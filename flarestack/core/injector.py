import os
import numpy as np
import healpy as hp
from flarestack.shared import k_to_flux, scale_shortener, band_mask_cache_name
from flarestack.core.energy_PDFs import EnergyPDF, read_e_pdf_dict
from flarestack.core.time_PDFs import TimePDF, read_t_pdf_dict
from flarestack.utils.dataset_loader import data_loader
from scipy import sparse


def read_injector_dict(inj_dict):
    """Ensures that injection dictionaries remain backwards-compatible

    :param inj_dict: Injection Dictionary
    :return: Injection Dictionary compatible with new format
    """

    maps = [
        ("Injection Time PDF", "injection_time_pdf"),
        ("Injection Energy PDF", "injection_energy_pdf"),
        ("Poisson Smear?", "poisson_smear_bool")
    ]

    for (old_key, new_key) in maps:

        if old_key in inj_dict.keys():
            inj_dict[new_key] = inj_dict[old_key]

    pairs = [
        ("injection_energy_pdf", read_e_pdf_dict),
        ("injection_time_pdf", read_t_pdf_dict)
    ]

    for (key, f) in pairs:
        if key in inj_dict.keys():
            inj_dict[key] = f(inj_dict[key])

    return inj_dict


class Injector:
    """Core Injector Class, returns a dataset on which calculations can be
    performed.
    """

    def __init__(self, season, sources, **kwargs):

        kwargs = read_injector_dict(kwargs)

        print "Initialising Injector for", season["Name"]
        self.injection_band_mask = dict()
        self.season = season

        self._raw_data = data_loader(season["exp_path"])

        self._mc = data_loader(season["mc_path"])

        self.sources = sources
        self.dist_scale = np.sum(self.sources["distance_mpc"]**-2)

        try:
            self.time_pdf = TimePDF.create(kwargs["injection_time_pdf"],
                                           season)
            self.energy_pdf = EnergyPDF.create(kwargs["injection_energy_pdf"])
            self.mc_weights = self.energy_pdf.weight_mc(self._mc)
        except KeyError:
            print "No Injection Arguments. Are you unblinding?"
            pass

        if "poisson_smear_bool" in kwargs.keys():
            self.poisson_smear = kwargs["poisson_smear_bool"]
        else:
            self.poisson_smear = True

        self.ref_fluxes = {
            scale_shortener(0.0): dict()
        }
        for source in sources:
            self.ref_fluxes[scale_shortener(0.0)][source["source_name"]] = 0.0

        try:
            self.fixed_n = kwargs["fixed_n"]
        except KeyError:
            self.fixed_n = np.nan

    def scramble_data(self):
        """Scrambles the raw dataset to "blind" the data. Assigns a flat Right
        Ascension distribution, and randomly redistributes the arrival times
        in the dataset. Returns a shuffled dataset, which can be used for
        blinded analysis.

        :return: data: The scrambled dataset
        """
        data = np.copy(self._raw_data)
        # Assigns a flat random distribution for Right Ascension
        data['ra'] = np.random.uniform(0, 2 * np.pi, size=len(data))
        # Randomly reorders the times
        np.random.shuffle(data["time"])

        return data

    def select_mc_band(self, source):
        """For a given source, selects MC events within a declination band of
        width +/- 5 degrees that contains the source. Then returns the MC data
        subset containing only those MC events.

        :param source: Source to be simulated
        :return: mc (cut): Simulated events which lie within the band
        :return: omega: Solid Angle of the chosen band
        :return: band_mask: The mask which removes events outside band
        """

        # Sets half width of band
        dec_width = np.deg2rad(5.)

        # Sets a declination band 5 degrees above and below the source
        min_dec = max(-np.pi / 2., source['dec_rad'] - dec_width)
        max_dec = min(np.pi / 2., source['dec_rad'] + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

        # Checks if the mask has already been evaluated for the source
        # If not, creates the mask for this source, and saves it
        if source["source_name"] in self.injection_band_mask.keys():
            band_mask = self.injection_band_mask[source["source_name"]]
        else:
            band_mask = np.logical_and(np.greater(self._mc["trueDec"], min_dec),
                                       np.less(self._mc["trueDec"], max_dec))
            self.injection_band_mask[source["source_name"]] = band_mask

        return np.copy(self._mc[band_mask]), omega, band_mask

    def calculate_single_source(self, source, scale):
        """Calculate the weighted MC for a signle source, given a flux scale
        and a distance scale.

        :param source:
        :param scale:
        :return:
        """
        # Selects MC events lying in a +/- 5 degree declination band
        source_mc, omega, band_mask = self.select_mc_band(source)

        # Calculate the effective injection time for simulation. Equal to
        # the overlap between the season and the injection time PDF for
        # the source, scaled if the injection PDF is not uniform in time.
        eff_inj_time = self.time_pdf.effective_injection_time(source)

        # All injection fluxes are given in terms of k, equal to 1e-9
        inj_flux = k_to_flux(source["injection_weight_modifier"] * scale)

        # Fraction of total flux allocated to given source, assuming
        # standard candles with flux proportional to 1/d^2

        dist_weight = (source["distance_mpc"] ** -2) / self.dist_scale

        # Weight coming from relative strength of sources.

        base_weight = source["base_weight"]

        # Calculate the fluence, using the effective injection time.
        fluence = inj_flux * eff_inj_time * dist_weight * base_weight

        # Recalculates the oneweights to account for the declination
        # band, and the relative distance of the sources.
        # Multiplies by the fluence, to enable calculations of n_inj,
        # the expected number of injected events

        source_mc["ow"] = fluence * (self.mc_weights[band_mask] / omega)

        return source_mc

    def inject_signal(self, scale):
        """Randomly select simulated events from the Monte Carlo dataset to
        simulate a signal for each source. The source flux can be scaled by
        the scale parameter.

        :param scale: Ratio of Injected Flux to source flux.
        :return: Set of signal events for the given IC Season.
        """
        # Creates empty signal event array
        sig_events = np.empty((0, ), dtype=self._raw_data.dtype)

        n_tot_exp = 0

        scale_key = scale_shortener(scale)

        if scale_key not in self.ref_fluxes.keys():
            self.ref_fluxes[scale_key] = dict()

        # Loop over each source to be simulated
        for i, source in enumerate(self.sources):

            source_mc = self.calculate_single_source(source, scale)

            # If a number of neutrinos to inject is specified, use that.
            # Otherwise, inject based on the flux scale as normal.

            if not np.isnan(self.fixed_n):
                n_inj = int(self.fixed_n)
            else:
                n_inj = np.sum(source_mc["ow"])

            n_tot_exp += n_inj

            if source["source_name"] not in self.ref_fluxes[scale_key].keys():
                self.ref_fluxes[scale_key][source["source_name"]] = n_inj

            # Simulates poisson noise around the expectation value n_inj.
            if self.poisson_smear:
                n_s = np.random.poisson(n_inj)
            # If there is no poisson noise, rounds n_s down to nearest integer
            else:
                n_s = int(n_inj)

            #  If n_s = 0, skips simulation step.
            if n_s < 1:
                continue

            # Creates a normalised array of OneWeights
            p_select = source_mc['ow'] / np.sum(source_mc['ow'])

            # Creates an array with n_signal entries.
            # Each entry is a random integer between 0 and no. of sources.
            # The probability for each integer is equal to the OneWeight of
            # the corresponding source_path.
            ind = np.random.choice(len(source_mc['ow']), size=n_s, p=p_select)

            # Selects the sources corresponding to the random integer array
            sim_ev = source_mc[ind]

            # Rotates the Monte Carlo events onto the source_path
            sim_ev = self.rotate_to_source(
                sim_ev, source['ra_rad'], source['dec_rad']
            )

            # Generates times for each simulated event, drawing from the
            # Injector time PDF.

            sim_ev["time"] = self.time_pdf.simulate_times(source, n_s)

            # Joins the new events to the signal events
            sig_events = np.concatenate(
                (sig_events, sim_ev[list(self._raw_data.dtype.names)]))

        print "Injected", n_tot_exp, "events"

        return sig_events

    def create_dataset(self, scale, pull_corrector):
        """Create a dataset based on scrambled data for background, and Monte
        Carlo simulation for signal. Returns the composite dataset. The source
        flux can be scaled by the scale parameter.

        :param scale: Ratio of Injected Flux to source flux
        :return: Simulated dataset
        """
        bkg_events = self.scramble_data()

        if scale > 0.:
            sig_events = self.inject_signal(scale)
        else:
            sig_events = []

        if len(sig_events) > 0:
            simulated_data = np.concatenate((bkg_events, sig_events))
        else:
            simulated_data = bkg_events

        simulated_data = pull_corrector.pull_correct_static(simulated_data)

        return simulated_data

    def rotate(self, ra1, dec1, ra2, dec2, ra3, dec3):
        """Rotate ra1 and dec1 in a way that ra2 and dec2 will exactly map
        onto ra3 and dec3, respectively. All angles are treated as radians.
        Essentially rotates the events, so that they behave as if they were
        originally incident on the source.

        :param ra1: Event Right Ascension
        :param dec1: Event Declination
        :param ra2: True Event Right Ascension
        :param dec2: True Event Declination
        :param ra3: Source Right Ascension
        :param dec3: Source Declination
        :return: Returns new Right Ascensions and Declinations
        """
        # Turns Right Ascension/Declination into Azimuth/Zenith for healpy
        phi1 = ra1 - np.pi
        zen1 = np.pi/2. - dec1
        phi2 = ra2 - np.pi
        zen2 = np.pi/2. - dec2
        phi3 = ra3 - np.pi
        zen3 = np.pi/2. - dec3

        # Rotate each ra1 and dec1 towards the pole?
        x = np.array([hp.rotator.rotateDirection(
            hp.rotator.get_rotation_matrix((dp, -dz, 0.))[0], z, p)
            for z, p, dz, dp in zip(zen1, phi1, zen2, phi2)])

        # Rotate **all** these vectors towards ra3, dec3 (source_path)
        zen, phi = hp.rotator.rotateDirection(np.dot(
            hp.rotator.get_rotation_matrix((-phi3, 0, 0))[0],
            hp.rotator.get_rotation_matrix((0, zen3, 0.))[0]), x[:, 0], x[:, 1])

        dec = np.pi/2. - zen
        ra = phi + np.pi
        return np.atleast_1d(ra), np.atleast_1d(dec)

    def rotate_to_source(self, ev, ra, dec):
        """Modifies the events by reassigning the Right Ascension and
        Declination of the events. Rotates the events, so that they are
        distributed as if they originated from the source. Removes the
        additional Monte Carlo information from sampled events, so that they
        appear like regular data.

        The fields removed are:
            True Right Ascension,
            True Declination,
            True Energy,
            OneWeight

        :param ev: Events
        :param ra: Source Right Ascension (radians)
        :param dec: Source Declination (radians)
        :return: Events (modified)
        """
        names = ev.dtype.names

        # Rotates the events to lie on the source
        ev["ra"], rot_dec = self.rotate(ev["ra"], np.arcsin(ev["sinDec"]),
                                        ev["trueRa"], ev["trueDec"],
                                        ra, dec)

        if "dec" in names:
            ev["dec"] = rot_dec
        ev["sinDec"] = np.sin(rot_dec)

        # Deletes the Monte Carlo information from sampled events
        non_mc = [name for name in names
                  if name not in ["trueRa", "trueDec", "trueE", "ow"]]
        ev = ev[non_mc].copy()

        return ev

class LowMemoryInjector(Injector):
    """For large numbers of sources O(~100), saving MC masks becomes
    increasingly burdensome. As a solution, the LowMemoryInjector should be
    used instead. It will be slower if you inject neutrinos multiple times,
    but will have much more reasonable memory consumption.
    """

    def __init__(self, season, sources, **kwargs):
        Injector.__init__(self, season, sources, **kwargs)
        self.split_cats, self.injection_band_paths = band_mask_cache_name(
            season, self.sources
        )

        if np.sum([not os.path.isfile(x) for x in self.injection_band_paths])\
                > 0.:
            self.make_injection_band_mask()
        else:
            print "Loading injection band mask from", self.injection_band_paths

    def make_injection_band_mask(self):

        for j, cat in enumerate(self.split_cats):

            path = self.injection_band_paths[j]

            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass

            # Make mask
            injection_band_mask = sparse.lil_matrix((len(cat),
                                                     len(self._mc)), dtype=bool)
            for i, source in enumerate(cat):
                dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(source)
                band_mask = np.logical_and(np.greater(self._mc["trueDec"], min_dec),
                                           np.less(self._mc["trueDec"], max_dec))
                injection_band_mask[i,:] = band_mask
            injection_band_mask = injection_band_mask.tocsr()
            sparse.save_npz(path, injection_band_mask)

            del injection_band_mask

            print "Saving to", path

    @staticmethod              
    def get_dec_and_omega( source):
        # Sets half width of band
        dec_width = np.deg2rad(2.)

        # Sets a declination band 5 degrees above and below the source
        min_dec = max(-np.pi / 2., source['dec_rad'] - dec_width)
        max_dec = min(np.pi / 2., source['dec_rad'] + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))
        return dec_width, min_dec, max_dec, omega

    def select_mc_band(self, source):
        """For a given source, selects MC events within a declination band of
        width +/- 5 degrees that contains the source. Then returns the MC data
        subset containing only those MC events.

        :param mc: Monte Carlo simulation
        :param source: Source to be simulated
        :return: mc (cut): Simulated events which lie within the band
        :return: omega: Solid Angle of the chosen band
        :return: band_mask: The mask which removes events outside band
        """
        dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(source)
        band_mask = np.logical_and(np.greater(self._mc["trueDec"], min_dec),
                                   np.less(self._mc["trueDec"], max_dec))
        return np.copy(self._mc[band_mask]), omega, band_mask

    def inject_signal(self, scale):
        """Randomly select simulated events from the Monte Carlo dataset to
        simulate a signal for each source. The source flux can be scaled by
        the scale parameter.

        :param scale: Ratio of Injected Flux to source flux.
        :return: Set of signal events for the given IC Season.
        """
        # Creates empty signal event array
        sig_events = np.empty((0, ), dtype=self._raw_data.dtype)

        n_tot_exp = 0

        scale_key = scale_shortener(scale)

        if scale_key not in self.ref_fluxes.keys():
            self.ref_fluxes[scale_key] = dict()

        for j, path in enumerate(self.injection_band_paths):

            injection_band_mask = sparse.load_npz(path)

            # Loop over each source to be simulated
            for i, source in enumerate(self.split_cats[j]):

                dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(source)
                band_mask = injection_band_mask.getrow(i).toarray()[0]
                source_mc = mc[band_mask]

                # Selects MC events lying in a +/- 5 degree declination band
                # source_mc, omega, band_mask = self.select_mc_band(mc, source)

                # Calculate the effective injection time for simulation. Equal to
                # the overlap between the season and the injection time PDF for
                # the source, scaled if the injection PDF is not uniform in time.
                eff_inj_time = self.time_pdf.effective_injection_time(source)

                # All injection fluxes are given in terms of k, equal to 1e-9
                inj_flux = k_to_flux(
                    source["injection_weight_modifier"] * scale)

                # Fraction of total flux allocated to given source, assuming
                # standard candles with flux proportional to 1/d^2

                dist_weight = (source["distance_mpc"] ** -2) / self.dist_scale

                # Weight coming from relative strength of sources.

                base_weight = source["base_weight"]

                # Calculate the fluence, using the effective injection time.
                fluence = inj_flux * eff_inj_time * dist_weight * base_weight

                # Recalculates the oneweights to account for the declination
                # band, and the relative distance of the sources.
                # Multiplies by the fluence, to enable calculations of n_inj,
                # the expected number of injected events

                source_mc["ow"] = fluence * (self.mc_weights[band_mask] / omega)

                if np.isnan(self.fixed_n):

                    n_inj = np.sum(source_mc["ow"])

                    n_tot_exp += n_inj

                    if source["source_name"] not in self.ref_fluxes[scale_key].keys():
                        self.ref_fluxes[scale_key][
                            source["source_name"]] = n_inj

                    # Simulates poisson noise around the expectation value n_inj.
                    if self.poisson_smear:
                        n_s = np.random.poisson(n_inj)
                    # If there is no poisson noise, rounds n_s down to nearest integer
                    else:
                        n_s = int(n_inj)

                else:
                    n_s = int(self.fixed_n)

                    if source["source_name"] not in self.ref_fluxes[scale_key].keys():
                        self.ref_fluxes[scale_key][source["source_name"]] = n_s

                #  If n_s = 0, skips simulation step.
                if n_s < 1:
                    continue

                # Creates a normalised array of OneWeights
                p_select = source_mc['ow'] / np.sum(source_mc['ow'])

                # Creates an array with n_signal entries.
                # Each entry is a random integer between 0 and no. of sources.
                # The probability for each integer is equal to the OneWeight of
                # the corresponding source_path.
                ind = np.random.choice(len(source_mc['ow']), size=n_s, p=p_select)

                # Selects the sources corresponding to the random integer array
                sim_ev = source_mc[ind]

                # Rotates the Monte Carlo events onto the source_path
                sim_ev = self.rotate_to_source(
                    sim_ev, source['ra_rad'], source['dec_rad']
                )

                # Generates times for each simulated event, drawing from the
                # Injector time PDF.

                sim_ev["time"] = self.time_pdf.simulate_times(source, n_s)

                # Joins the new events to the signal events
                sig_events = np.concatenate(
                    (sig_events, sim_ev[list(self._raw_data.dtype.names)]))

            del injection_band_mask

        # print "Injecting", n_tot_exp
        # print max([x/n_tot_exp for x
        #            in self.ref_fluxes[scale_key].itervalues()])
        # raw_input("prompt")

        return sig_events


class SparseMatrixInjector(Injector):
    """For large numbers of sources O(~100), saving MC masks becomes
    increasingly burdensome. As a solution, the LowMemoryInjector should be
    used instead. It will be slower if you inject neutrinos multiple times,
    but will have much more reasonable memory consumption.
    """

    def select_mc_band(self, mc, source):
        """For a given source, selects MC events within a declination band of
        width +/- 5 degrees that contains the source. Then returns the MC data
        subset containing only those MC events.

        :param mc: Monte Carlo simulation
        :param source: Source to be simulated
        :return: mc (cut): Simulated events which lie within the band
        :return: omega: Solid Angle of the chosen band
        :return: band_mask: The mask which removes events outside band
        """

        from scipy import sparse

        # Sets half width of band
        dec_width = np.deg2rad(5.)

        # Sets a declination band 5 degrees above and below the source
        min_dec = max(-np.pi / 2., source['dec_rad'] - dec_width)
        max_dec = min(np.pi / 2., source['dec_rad'] + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

        # Checks if the mask has already been evaluated for the source
        # If not, creates the mask for this source, and saves it
        if source["source_name"] in self.injection_band_mask.keys():
            # convert sparse matrix back to nparray
            band_mask = self.injection_band_mask[source["source_name"]]
        else:
            band_mask = np.logical_and(np.greater(mc["trueDec"], min_dec),
                                       np.less(mc["trueDec"], max_dec))
            self.injection_band_mask[source['source_name']] = band_mask

        return mc[band_mask], omega, band_mask


class MockUnblindedInjector(Injector):
    """If the data is not really to be unblinded, then MockUnblindedInjector
    should be called. In this case, the create_dataset function simply returns
    one background scramble.
    """

    def __init__(self, season, sources=np.nan, **kwargs):
        self.season = season
        self._raw_data = data_loader(season["exp_path"])

    def create_dataset(self, scale, pull_corrector):
        """Returns a background scramble

        :return: Scrambled data
        """
        seed = int(123456)
        np.random.seed(seed)

        simulated_data = self.scramble_data()
        simulated_data = pull_corrector.pull_correct_static(simulated_data)

        return simulated_data


class TrueUnblindedInjector(Injector):
    """If the data is unblinded, then UnblindedInjector should be called. In
    this case, the create_dataset function simply returns the unblinded dataset.
    """
    def __init__(self, season, sources, **kwargs):
        self.season = season
        self._raw_data = data_loader(season["exp_path"])

    def create_dataset(self, scale, pull_corrector):
        return pull_corrector.pull_correct_static(self._raw_data)


# if __name__ == "__main__":
#     from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
#     data = ps_7year[0]
#     from flarestack.analyses.agn_cores.shared_agncores import agncores_cat_dir
#     cat = np.load(
#         agncores_cat_dir +
#         "radioloud_2rxs_noBL_2000brightest_srcs_weight1.npy"
#     )
#
#     LowMemoryInjector(data, cat)


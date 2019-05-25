from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import object
import os
import numpy as np
import healpy as hp
import random
from flarestack.shared import k_to_flux, scale_shortener, band_mask_cache_name
from flarestack.core.energy_pdf import EnergyPDF, read_e_pdf_dict
from flarestack.core.time_pdf import TimePDF, read_t_pdf_dict
from flarestack.core.spatial_pdf import SpatialPDF
from flarestack.icecube_utils.dataset_loader import data_loader
from flarestack.utils.catalogue_loader import calculate_source_weight
from scipy import sparse, interpolate
from flarestack.shared import k_to_flux


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

        if old_key in list(inj_dict.keys()):
            inj_dict[new_key] = inj_dict[old_key]

    pairs = [
        ("injection_energy_pdf", read_e_pdf_dict),
        ("injection_time_pdf", read_t_pdf_dict)
    ]

    for (key, f) in pairs:
        if key in list(inj_dict.keys()):
            inj_dict[key] = f(inj_dict[key])

    if "injection_spatial_pdf" not in inj_dict.keys():
        inj_dict["injection_spatial_pdf"] = {}

    return inj_dict


class BaseInjector:
    """Base Injector Class
    """

    subclasses = {}

    def __init__(self, season, sources, **kwargs):

        kwargs = read_injector_dict(kwargs)
        self.inj_kwargs = kwargs

        print("Initialising Injector for", season.season_name)
        self.injection_band_mask = dict()
        self.season = season
        self.season.load_background_data()

        self.sources = sources

        if len(sources) > 0:
            self.weight_scale = calculate_source_weight(self.sources)

        try:
            self.time_pdf = TimePDF.create(kwargs["injection_time_pdf"],
                                           season)
            self.energy_pdf = EnergyPDF.create(kwargs["injection_energy_pdf"])
            self.spatial_pdf = SpatialPDF.create(
                kwargs["injection_spatial_pdf"])
        except KeyError:
            raise Exception("Injection Arguments missing. \n "
                            "'injection_energy_pdf', 'injection_time_pdf',"
                            "and 'injection_spatial_pdf' are required. \n"
                            "Found: \n {0}".format(kwargs))

        if "poisson_smear_bool" in list(kwargs.keys()):
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

    def update_sources(self, sources):
        """Reuses an injector with new sources

        :param sources: Sources to be added
        """
        self.sources = sources
        self.weight_scale = np.sum(
                self.sources["base_weight"] * self.sources["distance_mpc"]**-2)
        self.ref_fluxes = {
            scale_shortener(0.0): dict()
        }
        for source in sources:
            self.ref_fluxes[scale_shortener(0.0)][source["source_name"]] = 0.0

    def create_dataset(self, scale, pull_corrector):
        """Create a dataset based on scrambled data for background, and Monte
        Carlo simulation for signal. Returns the composite dataset. The source
        flux can be scaled by the scale parameter.

        :param scale: Ratio of Injected Flux to source flux
        :return: Simulated dataset
        """
        bkg_events = self.season.pseudo_background()

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

    def inject_signal(self, scale):
        return


    @classmethod
    def register_subclass(cls, inj_name):
        """Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """
        def decorator(subclass):
            cls.subclasses[inj_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, season, sources, **kwargs):

        inj_dict = read_injector_dict(kwargs)

        if "injector_name" not in inj_dict.keys():
            return cls(season, sources, **inj_dict)

        inj_name = inj_dict["injector_name"]

        if inj_name not in cls.subclasses:
            raise ValueError('Bad Injector name {}'.format(inj_name))
        else:
            return cls.subclasses[inj_name](season, sources, **inj_dict)


class MCInjector(BaseInjector):
    """Core Injector Class, returns a dataset on which calculations can be
    performed. This base class is tailored for injection of MC into mock
    background. This can be either MC background, or scrambled real data.
    """

    subclasses = {}

    def __init__(self, season, sources, **kwargs):
        kwargs = read_injector_dict(kwargs)
        BaseInjector.__init__(self, season, sources, **kwargs)

        self._mc = season.get_mc()

        try:
            self.mc_weights = self.energy_pdf.weight_mc(self._mc)
        except KeyError:
            print("No Injection Arguments. Are you unblinding?")
            pass

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
        if source["source_name"] in list(self.injection_band_mask.keys()):
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

        source_mc = self.calculate_fluence(source, scale, source_mc,
                                           band_mask, omega)

        return source_mc

    def calculate_fluence(self, source, scale, source_mc, band_mask, omega):
        """Function to calculate the fluence for a given source, and multiply
        the oneweights by this. After this step, the oneweight sum is equal
        to the expected neutrino number.

        :param source: Source to be calculated
        :param scale: Flux scale
        :param source_mc: MC that is close to source
        :param band_mask: Closeness mask for MC
        :param omega: Solid angle covered by MC mask
        :return: Modified source MC
        """
        # Calculate the effective injection time for simulation. Equal to
        # the overlap between the season and the injection time PDF for
        # the source, scaled if the injection PDF is not uniform in time.
        eff_inj_time = self.time_pdf.effective_injection_time(source)

        # All injection fluxes are given in terms of k, equal to 1e-9
        inj_flux = k_to_flux(source["injection_weight_modifier"] * scale)

        # Fraction of total flux allocated to given source, assuming
        # standard candles with flux proportional to 1/d^2 multiplied by the
        # sources weight

        weight = calculate_source_weight(source) / self.weight_scale

        # Calculate the fluence, using the effective injection time.
        fluence = inj_flux * eff_inj_time * weight

        # Recalculates the oneweights to account for the declination
        # band, and the relative distance of the sources.
        # Multiplies by the fluence, to enable calculations of n_inj,
        # the expected number of injected events

        source_mc["ow"] = fluence * self.mc_weights[band_mask] / omega

        return source_mc

    def inject_signal(self, scale):
        """Randomly select simulated events from the Monte Carlo dataset to
        simulate a signal for each source. The source flux can be scaled by
        the scale parameter.

        :param scale: Ratio of Injected Flux to source flux.
        :return: Set of signal events for the given IC Season.
        """
        # Creates empty signal event array
        sig_events = np.empty((0, ), dtype=self.season.get_background_dtype())

        n_tot_exp = 0

        scale_key = scale_shortener(scale)

        if scale_key not in list(self.ref_fluxes.keys()):
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

            if source["source_name"] not in list(self.ref_fluxes[scale_key].keys()):
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
            sim_ev = self.spatial_pdf.rotate_to_position(
                sim_ev, source['ra_rad'], source['dec_rad']
            )

            # Generates times for each simulated event, drawing from the
            # Injector time PDF.

            sim_ev["time"] = self.time_pdf.simulate_times(source, n_s)

            # Joins the new events to the signal events
            sig_events = np.concatenate(
                (sig_events,
                 sim_ev[list(self.season.get_background_dtype().names)])
            )

        return sig_events


@MCInjector.register_subclass("low_memory_injector")
class LowMemoryInjector(MCInjector):
    """For large numbers of sources O(~100), saving MC masks becomes
    increasingly burdensome. As a solution, the LowMemoryInjector should be
    used instead. It will be somewhat slower if you inject neutrinos multiple
    times, but will have much more reasonable memory consumption.
    """

    def __init__(self, season, sources, **kwargs):
        MCInjector.__init__(self, season, sources, **kwargs)
        self.split_cats, self.injection_band_paths = band_mask_cache_name(
            season, self.sources
        )

        if np.sum([not os.path.isfile(x) for x in self.injection_band_paths])\
                > 0.:
            self.make_injection_band_mask()
        else:
            print("Loading injection band mask from", self.injection_band_paths)

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

            print("Saving to", path)

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
        sig_events = np.empty((0, ), dtype=self.season.get_background_dtype())

        n_tot_exp = 0

        scale_key = scale_shortener(scale)

        if scale_key not in list(self.ref_fluxes.keys()):
            self.ref_fluxes[scale_key] = dict()

        for j, path in enumerate(self.injection_band_paths):

            injection_band_mask = sparse.load_npz(path)

            # Loop over each source to be simulated
            for i, source in enumerate(self.split_cats[j]):

                dec_width, min_dec, max_dec, omega = self.get_dec_and_omega(source)
                band_mask = injection_band_mask.getrow(i).toarray()[0]
                source_mc = np.copy(self._mc[band_mask])

                source_mc = self.calculate_fluence(source, scale, source_mc,
                                                   band_mask, omega)

                # If a number of neutrinos to inject is specified, use that.
                # Otherwise, inject based on the flux scale as normal.

                if not np.isnan(self.fixed_n):
                    n_inj = int(self.fixed_n)
                else:
                    n_inj = np.sum(source_mc["ow"])

                n_tot_exp += n_inj

                if source["source_name"] not in list(
                        self.ref_fluxes[scale_key].keys()):
                    self.ref_fluxes[scale_key][source["source_name"]] = n_inj

                # Simulates poisson noise around the expectation value n_inj.
                if self.poisson_smear:
                    n_s = np.random.poisson(n_inj)

                # If there is no poisson noise, rounds n_s to nearest integer
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

            del injection_band_mask

        # print "Injecting", n_tot_exp
        # print max([x/n_tot_exp for x
        #            in self.ref_fluxes[scale_key].itervalues()])
        # raw_input("prompt")

        return sig_events


class EffectiveAreaInjector(BaseInjector):
    """Class for injecting signal events by relying on effective areas rather
    than pre-existing Monte Carlo simulation. This Injector should be used
    for analysing public data, as no MC is provided.
    """

    def __init__(self, season, sources, **kwargs):
        BaseInjector.__init__(self, season, sources, **kwargs)
        self.effective_area_f = season.load_effective_area()
        self.energy_proxy_mapping = season.load_energy_proxy_mapping()
        self.angular_res_f = season.load_angular_resolution()
        self.conversion_cache = dict()

    def inject_signal(self, scale):

        # Creates empty signal event array
        sig_events = np.empty((0,),
                              dtype=self.season.get_background_dtype())

        n_tot_exp = 0

        scale_key = scale_shortener(scale)

        if scale_key not in list(self.ref_fluxes.keys()):
            self.ref_fluxes[scale_key] = dict()

        # Loop over each source to be simulated
        for i, source in enumerate(self.sources):

            # If a number of neutrinos to inject is specified, use that.
            # Otherwise, inject based on the flux scale as normal.

            if not np.isnan(self.fixed_n):
                n_inj = int(self.fixed_n)
            else:
                n_inj = self.calculate_single_source(source, scale)

            n_tot_exp += n_inj

            if source["source_name"] not in list(
                    self.ref_fluxes[scale_key].keys()):
                self.ref_fluxes[scale_key][source["source_name"]] = n_inj

            # Simulates poisson noise around the expectation value n_inj.
            if self.poisson_smear:
                n_s = np.random.poisson(n_inj)
            # If there is no poisson noise, rounds n_s to nearest integer
            else:
                n_s = int(n_inj)

            #  If n_s = 0, skips simulation step.
            if n_s < 1:
                continue

            sim_ev = np.empty(
                (n_s,), dtype=self.season.get_background_dtype())

            # Fills the energy proxy conversion cache

            if source["source_name"] not in self.conversion_cache.keys():
                self.calculate_energy_proxy(source)

            # Produces random seeds, converts this using a convolution
            # of energy pdf and approximated energy proxy mapping to
            # produce final energy proxy values

            convert_f = self.conversion_cache[source["source_name"]]

            random_fraction = [random.random() for _ in range(n_s)]

            sim_ev["logE"] = np.log10(np.exp(convert_f(random_fraction)))

            # Simulates times according to Time PDF

            sim_ev["time"] = self.time_pdf.simulate_times(source, n_s)
            sim_ev["sigma"] = self.angular_res_f(sim_ev["logE"]).copy()
            sim_ev["raw_sigma"] = sim_ev["sigma"].copy()

            sim_ev = self.spatial_pdf.simulate_distribution(source, sim_ev)

            sim_ev = sim_ev[list(
                self.season.get_background_dtype().names)].copy()
            #

            # Joins the new events to the signal events
            sig_events = np.concatenate((sig_events, sim_ev))

        sig_events = np.array(sig_events)

        return sig_events

    def calculate_single_source(self, source, scale):

        # Calculate the effective injection time for simulation. Equal to
        # the overlap between the season and the injection time PDF for
        # the source, scaled if the injection PDF is not uniform in time.
        eff_inj_time = self.time_pdf.effective_injection_time(source)

        # All injection fluxes are given in terms of k, equal to 1e-9
        inj_flux = k_to_flux(source["injection_weight_modifier"] * scale)

        # Fraction of total flux allocated to given source, assuming
        # standard candles with flux proportional to 1/d^2 multiplied by the
        # sources weight

        weight = calculate_source_weight(source) / self.weight_scale

        # Calculate the fluence, using the effective injection time.
        fluence = inj_flux * eff_inj_time * weight

        def source_eff_area(e):
            return self.effective_area_f(
                np.log10(e), np.sin(source["dec_rad"])) * self.energy_pdf.f(e)

        int_eff_a = self.energy_pdf.integrate_over_E(source_eff_area)

        # Effective areas are given in m2, but flux is in per cm2

        int_eff_a *= 10**4

        n_inj = fluence*int_eff_a

        return n_inj

    def calculate_energy_proxy(self, source):
        # Simulates energy proxy values

        def source_eff_area(log_e):
            return self.effective_area_f(log_e,
                                         np.sin(source["dec_rad"])) * \
                   self.energy_pdf.f(log_e) * \
                   self.energy_proxy_mapping(log_e)

        x_vals = np.linspace(
            np.log(self.energy_pdf.integral_e_min),
            np.log(self.energy_pdf.integral_e_max),
            100
        )[1:]

        y_vals = np.array([
            self.energy_pdf.integrate_over_E(source_eff_area, upper=np.exp(x))
            for x in x_vals
        ])
        y_vals /= max(y_vals)

        f = interpolate.interp1d(y_vals, x_vals)
        self.conversion_cache[source["source_name"]] = f


class MockUnblindedInjector:
    """If the data is not really to be unblinded, then MockUnblindedInjector
    should be called. In this case, the create_dataset function simply returns
    one background scramble.
    """

    def __init__(self, season, sources=np.nan, **kwargs):
        self.season = season
        self._raw_data = season.get_exp_data()

    def create_dataset(self, scale, pull_corrector):
        """Returns a background scramble

        :return: Scrambled data
        """
        seed = int(123456)
        np.random.seed(seed)

        simulated_data = self.season.pseudo_background()
        simulated_data = pull_corrector.pull_correct_static(simulated_data)

        return simulated_data


class TrueUnblindedInjector:
    """If the data is unblinded, then UnblindedInjector should be called. In
    this case, the create_dataset function simply returns the unblinded dataset.
    """
    def __init__(self, season, sources, **kwargs):
        self.season = season

    def create_dataset(self, scale, pull_corrector):
        return pull_corrector.pull_correct_static(self.season.get_exp_data())


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


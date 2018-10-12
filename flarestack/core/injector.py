import numpy as np
import healpy as hp
from flarestack.shared import k_to_flux, scale_shortener
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.core.time_PDFs import TimePDF
from flarestack.utils.dataset_loader import data_loader


class Injector:
    """Core Injector Class, returns a dataset on which calculations can be
    performed.
    """

    def __init__(self, season, sources, **kwargs):
        print "Initialising Injector for", season["Name"]
        self.injection_band_mask = dict()
        self.season = season

        self._raw_data = data_loader(season["exp_path"])

        self._mc = data_loader(season["mc_path"])

        self.sources = sources

        try:
            self.time_pdf = TimePDF.create(kwargs["Injection Time PDF"],
                                           season)
            self.energy_pdf = EnergyPDF.create(kwargs["Injection Energy PDF"])
            self.mc_weights = self.energy_pdf.weight_mc(self._mc)
        except KeyError:
            print "No Injection Arguments. Are you unblinding?"
            pass

        if "Poisson Smear?" in kwargs.keys():
            self.poisson_smear = kwargs["Poisson Smear?"]
        else:
            self.poisson_smear = True

        self.ref_fluxes = {
            scale_shortener(0.0): dict()
        }
        for source in sources:
            self.ref_fluxes[scale_shortener(0.0)][source["Name"]] = 0.0

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
        np.random.shuffle(data[self.season["MJD Time Key"]])

        return data

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

        # Sets half width of band
        dec_width = np.deg2rad(5.)

        # Sets a declination band 5 degrees above and below the source
        min_dec = max(-np.pi / 2., source['dec'] - dec_width)
        max_dec = min(np.pi / 2., source['dec'] + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

        # Checks if the mask has already been evaluated for the source
        # If not, creates the mask for this source, and saves it
        if source['Name'] in self.injection_band_mask.keys():
            band_mask = self.injection_band_mask[source['Name']]
        else:
            band_mask = np.logical_and(np.greater(mc["trueDec"], min_dec),
                                       np.less(mc["trueDec"], max_dec))
            self.injection_band_mask[source['Name']] = band_mask

        return mc[band_mask], omega, band_mask

    def inject_signal(self, scale):
        """Randomly select simulated events from the Monte Carlo dataset to
        simulate a signal for each source. The source flux can be scaled by
        the scale parameter.

        :param scale: Ratio of Injected Flux to source flux.
        :return: Set of signal events for the given IC Season.
        """
        mc = self._mc
        # Creates empty signal event array
        sig_events = np.empty((0, ), dtype=self._raw_data.dtype)

        n_tot_exp = 0

        dist_scale = np.sum(self.sources["Distance (Mpc)"]**-2)

        scale_key = scale_shortener(scale)

        if scale_key not in self.ref_fluxes.keys():
            self.ref_fluxes[scale_key] = dict()

        # Loop over each source to be simulated
        for i, source in enumerate(
                np.sort(self.sources, order="Distance (Mpc)")):

            # Calculate the effective injection time for simulation. Equal to
            # the overlap between the season and the injection time PDF for
            # the source, scaled if the injection PDF is not uniform in time.
            eff_inj_time = self.time_pdf.effective_injection_time(source)

            # Selects MC events lying in a +/- 5 degree declination band
            source_mc, omega, band_mask = self.select_mc_band(mc, source)

            # All injection fluxes are given in terms of k, equal to 1e-9
            inj_flux = k_to_flux(source['Relative Injection Weight'] * scale)

            # dist_weight = (source["Distance (Mpc)"]/dist_scale)**-2

            dist_weight = (source["Distance (Mpc)"]**-2) / dist_scale

            # Calculate the fluence, using the effective injection time.
            fluence = inj_flux * eff_inj_time * dist_weight

            # Recalculates the oneweights to account for the declination
            # band, and the relative distance of the sources.
            # Multiplies by the fluence, to enable calculations of n_inj,
            # the expected number of injected events

            source_mc["ow"] = fluence * (self.mc_weights[band_mask] / omega)

            n_inj = np.sum(source_mc["ow"])

            n_tot_exp += n_inj

            if source["Name"] not in self.ref_fluxes[scale_key].keys():
                self.ref_fluxes[scale_key][source["Name"]] = n_inj

            # print self.season["Name"], source["Name"], "expecting", n_inj,

            # Simulates poisson noise around the expectation value n_inj.
            if self.poisson_smear:
                n_s = np.random.poisson(n_inj)
            # If there is no poisson noise, rounds n_s down to nearest integer
            else:
                n_s = int(n_inj)

            # print "injecting", n_s
            # print inj_flux, dist_weight, eff_inj_time/(60*60*24), np.sum(
            #     self.mc_weights[ band_mask] / omega)
            # raw_input("prompt")

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
            sim_ev = self.rotate_to_source(sim_ev, source['ra'], source['dec'])
            # sim_ev = np.array(sim_ev, dtype=self._raw_data.dtype)

            # Generates times for each simulated event, drawing from the
            # Injector time PDF.

            sim_ev[self.season["MJD Time Key"]] = \
                self.time_pdf.simulate_times(source, n_s)

            # Joins the new events to the signal events
            sig_events = np.concatenate(
                (sig_events, sim_ev[list(self._raw_data.dtype.names)]))

        return sig_events

    def create_dataset(self, scale):
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
        :param ra: Source Right Ascension
        :param dec: Source Declination
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


class MockUnblindedInjector(Injector):
    """If the data is not really to be unblinded, then MockUnblindedInjector
    should be called. In this case, the create_dataset function simply returns
    one background scramble.
    """

    def __init__(self, season, sources, **kwargs):
        self.season = season
        self._raw_data = data_loader(season["exp_path"])

    def create_dataset(self, scale):
        """Returns a background scramble

        :return: Scrambled data
        """

        seed = int(123456)
        np.random.seed(seed)

        return self.scramble_data()


class TrueUnblindedInjector(Injector):
    """If the data is unblinded, then UnblindedInjector should be called. In
    this case, the create_dataset function simply returns the unblinded dataset.
    """
    def __init__(self, season, sources, **kwargs):
        self.season = season
        self._raw_data = data_loader(season["exp_path"])

    def create_dataset(self, scale):

        return self._raw_data

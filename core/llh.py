import numexpr
import astro
import numpy as np
import scipy.interpolate
import cPickle as Pickle
from signal_over_background import SoB
from shared import acceptance_path
from time_PDFs import TimePDF


class LLH(SoB):
    """General  LLH class.
    """

    def __init__(self, season, sources, splines=None, **kwargs):
        # print "Initialising LLH for", season["Name"]
        SoB.__init__(self, season, splines, **kwargs)
        self.sources = sources

        # If a time PDF is to be used, a dictionary must be provided in kwargs
        time_dict = kwargs["LLH Time PDF"]
        if time_dict is not None:
            self.time_pdf = TimePDF.create(time_dict, season)

        self.acceptance_f = self.create_acceptance_function()

    # def create_acceptance_function(self):
    #     """Creates a 2D linear interpolation of the acceptance of the detector
    #     for the given season, as a function of declination and gamma. Returns
    #     this interpolation function.
    #
    #     :return: 2D linear interpolation
    #     """
    #     dec_bins = np.load(
    #         self.season['aw_path'] + '_bins_dec.npy')
    #     gamma_bins = np.load(
    #         self.season['aw_path'] + '_bins_gamma.npy')
    #     values = np.load(self.season['aw_path'] + '_values.npy')
    #     f = scipy.interpolate.interp2d(
    #         dec_bins, gamma_bins, values, kind='linear')
    #     return f

    def create_acceptance_function(self):
        """Creates a 2D linear interpolation of the acceptance of the detector
        for the given season, as a function of declination and gamma. Returns
        this interpolation function.

        :return: 2D linear interpolation
        """

        acc_path = acceptance_path(self.season["Name"])

        with open(acc_path) as f:
            acc_dict = Pickle.load(f)

        dec_bins = acc_dict["dec"]
        gamma_bins = acc_dict["gamma"]
        values = acc_dict["acceptance"]
        f = scipy.interpolate.interp2d(
            dec_bins, gamma_bins, values.T, kind='linear')
        return f


    def acceptance(self, source, params=None):
        """Calculates the detector acceptance for a given source, using the
        2D interpolation of the acceptance as a function of declination and
        gamma. If gamma IS NOT being fit, uses the default value of gamma for
        weighting (determined in __init__). If gamma IS being fit, it will be
        the last entry in the parameter array, and is the acceptance uses
        this value.

        :param source: Source to be considered
        :param params: Parameter array
        :return: Value for the acceptance of the detector, in the given
        season, for the source
        """
        dec = source["dec"]
        if not self.fit_gamma:
            gamma = self.default_gamma
        else:
            gamma = params[-1]

        return self.acceptance_f(dec, gamma)

    def select_spatially_coincident_data(self, data, sources):
        """Checks each source, and only identifies events in data which are
        both spatially and time-coincident with the source. Spatial
        coincidence is defined as a +/- 5 degree box centered on the  given
        source. Time coincidence is determined by the parameters of the LLH
        Time PDF. Produces a mask for the dataset, which removes all events
        which are not coincident with at least one source.

        :param data: Dataset to be tested
        :param sources: Sources to be tested
        :return: Mask to remove
        """
        veto = np.ones_like(data["timeMJD"], dtype=np.bool)

        # print min(data["timeMJD"]), max(data["timeMJD"])

        for source in sources:

            # Sets half width of spatial box
            width = np.deg2rad(5.)

            # Sets a declination band 5 degrees above and below the source
            min_dec = max(-np.pi / 2., source['dec'] - width)
            max_dec = min(np.pi / 2., source['dec'] + width)

            # Accepts events lying within a 5 degree band of the source
            dec_mask = np.logical_and(np.greater(data["dec"], min_dec),
                                      np.less(data["dec"], max_dec))

            # Sets the minimum value of cos(dec)
            cos_factor = np.amin(np.cos([min_dec, max_dec]))

            # Scales the width of the box in ra, to give a roughly constant
            # area. However, if the width would have to be greater that +/- pi,
            # then sets the area to be exactly 2 pi.
            dPhi = np.amin([2. * np.pi, 2. * width / cos_factor])

            # Accounts for wrapping effects at ra=0, calculates the distance
            # of each event to the source.
            ra_dist = np.fabs(
                (data["ra"] - source['ra'] + np.pi) % (2. * np.pi) - np.pi)
            ra_mask = ra_dist < dPhi / 2.

            spatial_mask = dec_mask & ra_mask

            veto = veto & ~spatial_mask

        # print min(data[~veto]["timeMJD"]), max(data[~veto]["timeMJD"])

        # print "Of", len(data), "total events, we consider", np.sum(~veto), \
        #     "events which are coincident with the sources."
        return ~veto

    def create_llh_function(self, data):
        n_all = float(len(data))
        SoB_spacetime = []

        if hasattr(self, "energy_pdf"):
            SoB_energy_cache = []
        # Otherwise, pass no energy weight information
        else:
            SoB_energy_cache = None

        n_mask = np.zeros(len(self.sources))

        for i, source in enumerate(self.sources):

            s_mask = self.select_spatially_coincident_data(data, [source])

            coincident_data = data[s_mask]
            n_mask[i] = np.sum(s_mask)

            if len(coincident_data) > 0:

                sig = self.signal_pdf(source, coincident_data)
                bkg = np.array(self.background_pdf(source, coincident_data))

                SoB_spacetime.append(sig/bkg)
                del sig
                del bkg

                # If an llh energy PDF has been provided, calculate the SoB values
                # for the coincident data, and stores it in a cache.
                if SoB_energy_cache is not None:
                    energy_cache = self.create_SoB_energy_cache(coincident_data)

                    # If gamma is not going to be fit, replaces the SoB energy
                    # cache with the weight array corresponding to the gamma provided
                    # in the llh energy PDF
                    if not self.fit_gamma:
                        energy_cache = self.estimate_energy_weights(
                            self.default_gamma, energy_cache)

                    SoB_energy_cache.append(energy_cache)

        def test_statistic(params, weights):
            return self.calculate_test_statistic(
                params, weights, n_all, SoB_spacetime,
                SoB_energy_cache)

        return test_statistic

    def calculate_test_statistic(self, params, weights,
                                 n_all, SoB_spacetime,
                                 SoB_energy_cache=None):
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from minimisation
        :return: Test Statistic
        """

        # If fitting gamma and calculates the energy weights for the given
        # value of gamma
        if self.fit_gamma:
            n_s = np.array(params[:-1])
            gamma = params[-1]

            SoB_energy = np.array([self.estimate_energy_weights(gamma, x) for x
                                in SoB_energy_cache])

            # SoB_energy = [self.estimate_energy_weights(gamma,
            #                                            SoB_energy_cache[0])]

        # If using energy information but with a fixed value of gamma,
        # sets the weights as equal to those for the provided gamma value.
        elif SoB_energy_cache is not None:
            n_s = np.array(params)
            SoB_energy = np.array(SoB_energy_cache)

        # If not using energy information, assigns a weight of 1. to each event
        else:
            n_s = np.array(params)
            SoB_energy = np.array(1.)

        # Calculates the expected number of signal events for each source in
        # the season
        all_n_j = (n_s * weights).T


        # llh_value = 0
        #
        # for i, SoB_e in enumerate(SoB_energy):
        #     n_mask = float(len(SoB_e))
        #     # SoB = SoB_energy[i] * SoB_spacetime[i]
        #     #
        #     # chi = (SoB - 1)/n_all[i]
        #
        #     # print i, SoB_e, SoB_spacetime[i], n_all[i], n_mask[i], chi
        #     #
        #     # print np.sum(np.log1p(all_n_j * chi))
        #
        #     # Evaluate the likelihood function for neutrinos close to each source
        #     llh_value += np.sum(np.log1p(
        #         all_n_j.T[i] * ((SoB_energy[i] * SoB_spacetime[i]) -  1.)/n_all))
        #
        #     # Account for the events in the veto region, by assuming they have S/B=0
        #     llh_value += self.assume_background(all_n_j.T[i], n_mask, n_all)

        n_mask = np.array([len(x) for x in SoB_spacetime])

        x = 1. + ((all_n_j/n_all) * (SoB_energy * np.array(SoB_spacetime) -
                                   1.))[0]

        llh_value = np.sum([np.log(y) for y in x])

        llh_value += self.assume_background(all_n_j, n_mask, n_all)

        # Definition of test statistic
        return 2. * llh_value

# ==============================================================================
# Signal PDF
# ==============================================================================

    def signal_pdf(self, source, cut_data):
        """Calculates the value of the signal spatial PDF for a given source
        for each event in the coincident data subsample. If there is a Time PDF
        given, also calculates the value of the signal Time PDF for each event.
        Returns either the signal spatial PDF values, or the product of the
        signal spatial and time PDFs.

        :param source: Source to be considered
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Signal Spacetime PDF values
        """
        space_term = self.signal_spatial(source, cut_data)

        # print space_term,

        if hasattr(self, "time_pdf"):
            time_term = self.time_pdf.signal_f(cut_data["timeMJD"], source)
            sig_pdf = space_term * time_term

        else:
            sig_pdf = space_term

        # print sig_pdf
        # raw_input("prompt")

        return sig_pdf

    def signal_spatial(self, source, cut_data):
        """Calculates the angular distance between the source and the
        coincident dataset. Uses a Gaussian PDF function, centered on the
        source. Returns the value of the Gaussian at the given distances.

        :param source: Single Source
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Spatial PDF values
        """
        distance = astro.angular_distance(
            cut_data['ra'], cut_data['dec'], source['ra'], source['dec'])
        space_term = (1. / (2. * np.pi * cut_data['sigma'] ** 2.) *
                      np.exp(-0.5 * (distance / cut_data['sigma']) ** 2.))
        return space_term

# ==============================================================================
# Background PDF
# ==============================================================================

    def background_pdf(self, source, cut_data):
        """Calculates the value of the background spatial PDF for a given
        source for each event in the coincident data subsample. Thus is done
        by calling the self.bkg_spline spline function, which was fitted to
        the Sin(Declination) distribution of the data.

        If there is a signal Time PDF given, then the background time PDF
        is also calculated for each event. This is assumed to be a normalised
        uniform distribution for the season.

        Returns either the background spatial PDF values, or the product of the
        background spatial and time PDFs.

        :param source: Source to be considered
        :param cut_data: Subset of Dataset with coincident events
        :return: Array of Background Spacetime PDF values
        """
        space_term = (1. / (2. * np.pi)) * np.exp(
            self.bkg_spatial(cut_data["sinDec"]))

        if hasattr(self, "time_pdf"):
            time_term = self.time_pdf.background_f(cut_data["timeMJD"], source)
            sig_pdf = space_term * time_term
        else:
            sig_pdf = space_term

        return sig_pdf

# ==============================================================================
# Energy Log(Signal/Background) Ratio
# ==============================================================================

    def create_SoB_energy_cache(self, cut_data):
        """Evaluates the Log(Signal/Background) values for all coincident
        data. For each value of gamma in self.gamma_support_points, calculates
        the Log(Signal/Background) values for the coincident data. Then saves
        each weight array to a dictionary.

        :param cut_data: Subset of the data containing only coincident events
        :return: Dictionary containing SoB values for each event for each
        gamma value.
        """
        energy_SoB_cache = dict()

        for gamma in self.SoB_spline_2Ds.keys():
            energy_SoB_cache[gamma] = self.SoB_spline_2Ds[gamma].ev(
                cut_data["logE"], cut_data["sinDec"])

        return energy_SoB_cache

    def estimate_energy_weights(self, gamma, energy_SoB_cache):
        """Quickly estimates the value of Signal/Background for Gamma.
        Uses pre-calculated values for first and second derivatives.
        Uses a Taylor series to estimate S(gamma), unless SoB has already
        been calculated for a given gamma.

        :param gamma: Spectral Index
        :param energy_SoB_cache: Weight cache
        :return: Estimated value for S(gamma)
        """
        if gamma in energy_SoB_cache.keys():
            val = np.exp(energy_SoB_cache[gamma])
        else:
            g1 = self._around(gamma)
            dg = self.precision

            g0 = self._around(g1 - dg)
            g2 = self._around(g1 + dg)

            # Uses Numexpr to quickly estimate S(gamma)

            S0 = energy_SoB_cache[g0]
            S1 = energy_SoB_cache[g1]
            S2 = energy_SoB_cache[g2]

            # print "S0", S0[:5]
            # print "S1", S1[:5]
            # print "S2", S2[:5]

            val = numexpr.evaluate(
                "exp((S0 - 2.*S1 + S2) / (2. * dg**2) * (gamma - g1)**2" + \
                " + (S2 -S0) / (2. * dg) * (gamma - g1) + S1)"
            )

            # print np.log(val[:5])
            #
            # print g0, g1, g2, gamma
            #
            # raw_input("prompt")

        return val

    def assume_background(self, n_s, n_coincident, n_all):
        """To save time with likelihood calculation, it can be assumed that
        all events defined as "non-coincident", because of distance in space
        and time to the source, are in fact background events. This is
        equivalent to setting S=0 for all non-coincident events. IN this
        case, the likelihood can be calculated as the product of the number
        of non-coincident events, and the likelihood of an event which has S=0.

        :param n_s: Array of expected number of events
        :param n_coincident: Number of events that were not assumed to have S=0
        :param n_all: The total number of events
        :return: Log Likelihood value for the given
        """

        return np.sum((n_all - n_coincident) * np.log1p((-n_s / n_all)))


class FlareLLH(LLH):

    def create_llh_function(self, data, flare_veto, n_all,
                            marginilisation):

        coincident_data = data[~flare_veto]

        n_mask = len(coincident_data)

        sig = self.signal_pdf(self.sources, coincident_data)
        bkg = self.background_pdf(self.sources, coincident_data)
        SoB_spacetime = sig/bkg
        del sig
        del bkg

        # If an llh energy PDF has been provided, calculate the SoB values
        # for the coincident data, and stores it in a cache.
        if hasattr(self, "energy_pdf"):
            SoB_energy_cache = self.create_SoB_energy_cache(coincident_data)

            # If gamma is not going to be fit, replaces the SoB energy
            # cache with the weight array corresponding to the gamma provided
            # in the llh energy PDF
            if not self.fit_gamma:
                SoB_energy_cache = self.estimate_energy_weights(
                    self.default_gamma, SoB_energy_cache)

        else:
            SoB_energy_cache = None

        def test_statistic(params, weights):
            return self.calculate_flare_test_statistic(
                params, weights, n_all, marginilisation, SoB_spacetime,
                SoB_energy_cache)

        return test_statistic

    def calculate_flare_test_statistic(self, params, weights, n_all,
                                       marginilisation, SoB_spacetime,
                                       SoB_energy_cache=None):
        """Calculates the test statistic, given the parameters. Uses numexpr
        for faster calculations.

        :param params: Parameters from minimisation
        :return: Test Statistic
        """
        n_mask = len(SoB_spacetime)

        # If fitting gamma and calculates the energy weights for the given
        # value of gamma
        if self.fit_gamma:
            n_s = np.array(params[:-1])
            gamma = params[-1]

            SoB_energy = self.estimate_energy_weights(gamma, SoB_energy_cache)

        # If using energy information but with a fixed value of gamma,
        # sets the weights as equal to those for the provided gamma value.
        elif SoB_energy_cache is not None:
            n_s = np.array(params)
            SoB_energy = SoB_energy_cache

        # If not using energy information, assigns a weight of 1. to each event
        else:
            n_s = np.array(params)
            SoB_energy = 1.

        # Calculates the expected number of signal events for each source in
        # the season
        n_j = n_s * weights

        # Evaluate the likelihood function for neutrinos close to each source
        llh_value = np.sum(np.log((
            1 + ((n_j/n_all) * (SoB_energy * SoB_spacetime)))))

        # Account for the events in the veto region, by assuming they have S/B=0

        llh_value += self.assume_background(n_j, n_mask, n_all)

        # Definition of test statistic
        return 2. * llh_value

    def find_significant_events(self, coincident_data, sources):

        sig = self.signal_pdf(sources, coincident_data)
        bkg = self.background_pdf(sources, coincident_data)
        SoB_spacetime = sig / bkg

        SoB_energy_cache = self.create_SoB_energy_cache(coincident_data)

        SoB_energy = self.estimate_energy_weights(
                self.default_gamma, SoB_energy_cache)

        SoB = SoB_spacetime * SoB_energy

        mask = SoB > 1.

        return coincident_data[mask]

    def create_flare(self, season, sources, **kwargs):
        spline_dict = dict()
        spline_dict["Background spatial"] = self.bkg_spatial
        spline_dict["SoB_spline_2D"] = self.SoB_spline_2Ds

        flare_llh = FlareLLH(season, sources, spline_dict, **kwargs)
        return flare_llh
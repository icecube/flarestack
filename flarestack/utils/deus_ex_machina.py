"""Script used to estimate discovery potential quickly. It uses some pretty
absurd approximations, but is REALLY REALLY fast (equivalent or faster than a
single trial, and is good to within ~30-50%). Use with some caution to
quickly guess appropriate flux scales, or understand trends, without full
calculations.
"""
from __future__ import print_function
from __future__ import division
from builtins import object
import numpy as np
from flarestack.core.llh import LLH
from flarestack.core.injector import Injector
from flarestack.core.astro import angular_distance
from flarestack.shared import weighted_quantile, k_to_flux, flux_to_k
from flarestack.utils.catalogue_loader import load_catalogue, \
    calculate_source_weight
from scipy.stats import norm


def estimate_discovery_potential(injectors, sources):
    """Function to estimate discovery potential given an injection model. It
    assumes an optimal LLH construction, i.e aligned time windows and correct
    energy weighting etc. Takes injectors and seasons.

    :param injectors: Injectors to be used
    :param sources: Sources to be evaluated
    :return: An estimate for the discovery potential
    """
    print("Trying to guess scale!")

    season_bkg = []
    season_sig = []

    def weight_f(n_s, n_bkg):
        metric = np.array(n_s)/np.sqrt(np.array(n_bkg))
        return metric / np.mean(metric)

    weight_scale = calculate_source_weight(sources)

    livetime = 0.

    for (season, inj) in injectors.items():

        llh_dict = {"llh_name": "fixed_energy"}
        llh_dict["llh_energy_pdf"] = inj.inj_kwargs["injection_energy_pdf"]
        llh_dict["llh_time_pdf"] = inj.inj_kwargs["injection_time_pdf"]
        llh = LLH.create(inj.season, sources, llh_dict)

        # print("Season", season)
        data = inj._raw_data
        # n_bkg_tot += len(inj._raw_data)
        # print("Number of events", n_bkg_tot)
        livetime += inj.time_pdf.livetime * 60 * 60 * 24
        # print("Livetime is {0} seconds ({1} days)".format(
        #     livetime, inj.time_pdf.livetime
        # ))

        def signalness(sig_over_background):
            """Converts a signal over background ratio into a signal
            probability. This is ratio/(1 + ratio)

            :param sig_over_background: Ratio of signal to background
            probability
            :return: Percentage probability of signal
            """

            return sig_over_background / (1. + sig_over_background)

        n_sigs = []
        n_bkgs = []

        for source in sources:
            source_mc = inj.calculate_single_source(source, scale=1.0)

            # Sets half width of band
            dec_width = np.deg2rad(5.)

            # Sets a declination band 5 degrees above and below the source
            min_dec = max(-np.pi / 2., source['dec_rad'] - dec_width)
            max_dec = min(np.pi / 2., source['dec_rad'] + dec_width)
            # Gives the solid angle coverage of the sky for the band
            omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

            data_mask = np.logical_and(
                np.greater(data["dec"], min_dec),
                np.less(data["dec"], max_dec))
            local_data = data[data_mask]

            data_weights = signalness(llh.energy_weight_f(local_data))

            mc_weights = signalness(llh.energy_weight_f(source_mc))

            # The flux is split across sources. The source weight is equal to
            # the base source weight / source distance ^2. It is equal to
            # the fraction of total flux contributed by an individual source.

            source_weight = calculate_source_weight(source) / weight_scale

            # Assume we only count within the 50% containment for the source

            n_sig = 0.5 * np.sum(
                source_mc["ow"] * mc_weights)

            true_errors = angular_distance(
                source_mc["ra"], source_mc["dec"],
                source_mc["trueRa"], source_mc["trueDec"])

            median_sigma = weighted_quantile(
                        true_errors, 0.5, source_mc["ow"] * mc_weights)

            area = np.pi * median_sigma ** 2 / np.cos(source["dec_rad"])

            local_rate = np.sum(data_weights)

            n_bkg = local_rate * area #* source_weight

            sig_scale = 1.
            sig_scale = np.sqrt(np.mean(data_weights))
            # sig_scale = np.sqrt(n_bkg)
            # sig_scale = n_bkg/np.mean(data_weights)
            # "ow"]) #/
            # np.sqrt(
            # n_bkg)# + n_sig)

            n_sigs.append(n_sig / sig_scale)
            n_bkgs.append(n_bkg / sig_scale)

        n_sigs = np.array(n_sigs)
        n_bkgs = np.array(n_bkgs)

        weights = weight_f(n_sigs, n_bkgs)
        # weights = (n_sigs / np.mean(n_sigs)) #/ np.sqrt(float(len(sources))) #*
        # np.median(n_bkgs)

        sum_n_sigs = np.sum(n_sigs * weights)
        sum_n_bkgs = np.sum(n_bkgs * weights)

        season_sig.append(sum_n_sigs)
        season_bkg.append(sum_n_bkgs)

    season_sig = np.array(season_sig)
    season_bkg = np.array(season_bkg)

    season_weights = 1.

    int_sig = np.sum(season_sig * season_weights)
    int_bkg = np.sum(season_bkg * season_weights)

    disc_count = norm.ppf(norm.cdf(5.0), loc=int_bkg, scale=np.sqrt(int_bkg))

    disc_pot = disc_count - int_bkg

    scale = disc_pot / int_sig

    # The approximate scaling for idealised + binned to unbinned
    # Scales with high vs low statistics. For low statistics, you are
    # effectively are background free, so don't take the 50% hit for only
    # counting nearby neutrinos. In high-statics regime, previous study
    # showed ~factor of 2 improvement for binned vs unbinned

    # fudge_factor = (1.25 + 0.75 * np.tanh(np.log(int_bkg)))
    # fudge_factor = (1.25 + 0.75 * np.tanh(np.log(disc_count)))
    fudge_factor = 2.0
    fudge_factor *= 1.2

    scale /= fudge_factor

    # Convert from scale factor to flux units

    scale = k_to_flux(scale)

    print()
    print(
        "Estimated Discovery Potential is: {:.3g} GeV sr^-1 s^-1 cm^-2".format(
            scale
        ))
    print()
    return scale


class DeusExMachina(object):

    def __init__(self, seasons, inj_dict):

        self.seasons = seasons
        self.injectors = dict()

        for season in self.seasons:
            self.injectors[season["Name"]] = Injector(season, [], **inj_dict)

    def guess_discovery_potential(self, source_path):
        sources = load_catalogue(source_path)

        for inj in self.injectors.values():
            inj.update_sources(sources)

        return estimate_discovery_potential(self.injectors, sources)
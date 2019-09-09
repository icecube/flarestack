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
from flarestack.core.astro import angular_distance
from flarestack.shared import weighted_quantile, k_to_flux, flux_to_k
from flarestack.utils.catalogue_loader import load_catalogue, \
    calculate_source_weight
from scipy.stats import norm


def estimate_discovery_potential(injectors, sources, raw_scale=1.0):
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
        metric = np.array(n_s)#/np.sqrt(np.array(n_bkg))
        return metric / np.mean(metric)#/ max(metric)

    def ts_weight(n_s):
        return 1.
        # return n_s / np.sum(n_s)

    # def weight_ts(ts, n_s)

    weight_scale = calculate_source_weight(sources)

    livetime = 0.

    n_s_tot = 0.
    n_tot = 0.

    all_ns = []
    all_nbkg = []

    all_ts = []
    all_bkg_ts = []

    for (season, inj) in injectors.items():

        llh_dict = {"llh_name": "fixed_energy"}
        llh_dict["llh_energy_pdf"] = inj.inj_kwargs["injection_energy_pdf"]
        llh_dict["llh_sig_time_pdf"] = inj.inj_kwargs["injection_sig_time_pdf"]
        llh = LLH.create(inj.season, sources, llh_dict)

        # print("Season", season)
        data = inj.season.pseudo_background()
        n_tot += len(data)
        # n_bkg_tot += len(inj._raw_data)
        # print("Number of events", n_bkg_tot)
        livetime += inj.sig_time_pdf.livetime * 60 * 60 * 24
        # print("Livetime is {0} seconds ({1} days)".format(
        #     livetime, inj.sig_time_pdf.livetime
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

        ts_vals = []
        bkg_vals = []
        n_s_season = 0.

        n_exp = np.sum(inj.n_exp["n_exp"]) * raw_scale

        for source in sources:
            source_mc = inj.calculate_single_source(source, scale=raw_scale)

            # dist = angular_distance(
            #     source_mc["trueRa"], source_mc["trueDec"],
            #     source_mc["ra"], source_mc["dec"],
            # )/source_mc["sigma"]
            # #
            # # # # Only consider MC 3 sigma away or less
            # # #
            # min_dist = 3.
            # source_mc = source_mc[dist < min_dist]
            #
            # delta_rad = 0.5
            #
            # mask = np.logical_and(
            #     source_mc["trueRa"] < source["ra_rad"] + delta_rad,
            #     source_mc["trueRa"] > source["ra_rad"] - delta_rad,
            # )
            #
            # source_mc = source_mc[mask]
            #
            #
            # # Rotates the events to lie on the source
            # source_mc["ra"], rot_dec = llh.spatial_pdf.rotate(
            #     source_mc["ra"], np.arcsin(source_mc["sinDec"]),
            #     source_mc["trueRa"], source_mc["trueDec"],
            #     source["ra_rad"], source["dec_rad"])
            #
            # source_mc["dec"] = rot_dec
            # source_mc["sinDec"] = np.sin(rot_dec)
            # source_mc["trueRa"] = source["ra_rad"]
            # source_mc["trueDec"] = source["dec_rad"]

            # print(len(source_mc))
            #
            # print(np.mean(source_mc["ra"]), np.mean(source_mc["dec"]))
            # shifted_source_mc = llh.spatial_pdf.rotate_to_position(
            #     source_mc.copy(), source['ra_rad'], source['dec_rad']
            # ).copy()
            # source_mc["ra"] = shifted_source_mc["ra"].copy()
            # source_mc["dec"] = shifted_source_mc["dec"].copy()
            # print(np.mean(source_mc["ra"]))
            # print("Source:", source["ra_rad"])
            # print(np.mean(source_mc["dec"]))
            # print("?")

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

            # print("source_mc", source_mc)

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
            # sig_scale = np.sqrt(np.mean(data_weights))
            # sig_scale = np.sqrt(n_bkg)
            # sig_scale = n_bkg/np.mean(data_weights)
            # "ow"]) #/
            # np.sqrt(
            # n_bkg)# + n_sig)

            n_sigs.append(n_sig / sig_scale)
            n_bkgs.append(n_bkg / sig_scale)

            # print(min(true_errors), max(true_errors))

            # source_mc["sigma"] = 1.177 * true_errors
            #
            # print(max(llh.background(source_mc)),
            #       min(llh.background(source_mc)))
            # print(max(llh.spatial_pdf.signal(
            #     source, source_mc) / llh.background(source_mc)),
            #       min(llh.spatial_pdf.signal(
            #     source, source_mc) / llh.background(source_mc)))
            #
            # input("?")

            ratio_energy = llh.energy_weight_f(source_mc)

            ratio_time = livetime/llh.sig_time_pdf.effective_injection_time(
                source)

            bkg_energy = llh.energy_weight_f(local_data)

            # print(min(1. / (2. * np.pi * true_errors ** 2.) *
            #      np.exp(-0.5 * ((1.177) **
            #                     2.))))
            # print(min(llh.background(source_mc)),
            #       max(llh.background(source_mc)))
            #
            #
            # ratio_spatial = (1. / (2. * np.pi * source_mc["sigma"] ** 2.) *
            #                  np.exp(-0.5 * ((true_errors/source_mc["sigma"]) **
            #                     2.))) / llh.background(source_mc)

            # ratio_spatial = np.log(gaussian_spatial)
            # print("Spatial", np.mean(gaussian_spatial), np.mean(
            #     ratio_spatial*source_mc["ow"]/np.mean(source_mc["ow"])))
            #
            # input("?")

            sig_spatial = (1. / (2. * np.pi * source_mc["sigma"] ** 2.) *
                             np.exp(-0.5 * (
                                     (true_errors/source_mc["sigma"]) **2.))) \
                            / llh.spatial_pdf.background_spatial(source_mc)

            bkg_spatial = (1. / (2. * np.pi * local_data["sigma"] ** 2.) *
                             np.exp(-0.5 * (
                                     (1.177) ** 2.))) \
                            / llh.spatial_pdf.background_spatial(local_data)


            # print(min(ratio_spatial), np.mean(ratio_spatial), max(ratio_spatial))
            # print(np.mean((1. / (2. * np.pi * source_mc["sigma"] ** 2.) *
            #                  np.exp(-0.5 * (
            #                          (true_errors/source_mc["sigma"]) **2.)))))
            # input("???")

            ratio_spatial = sig_spatial

            # ratio_spatial = 1.

            dists = angular_distance(
                local_data["ra"], local_data["dec"],
                source["ra_rad"], source["dec_rad"])

            bkg_spatial = (1. / (2. * np.pi * local_data["sigma"] ** 2.) *
                           np.exp(-0.5 * ((dists/local_data["sigma"]) ** 2.))) \
                            / llh.spatial_pdf.background_spatial(local_data)

            # print(np.mean(np.log10(bkg_spatial)))
            # input("?")

            # bkg_spatial = 1.

            sum_ratio = np.array(ratio_energy * ratio_spatial * ratio_time)
            mask = sum_ratio > 0.

            n_s_tot += np.sum(source_mc["ow"])
            n_s_season += np.sum(source_mc["ow"])
            med_ts = np.sum(np.log(1. + sum_ratio / len(data))
                            * source_mc["ow"])

            # med_ts = np.sum(np.log(1 + sum_ratio[mask]) * source_mc["ow"][mask])

            # med_ts = np.sum(np.log(1 + (sum_ratio[mask] * source_mc[
            #     "ow"][mask]/len(local_data))))

            # print(np.sum(np.log(1. + sum_ratio/len(data))))
            # print(np.log(1 + np.mean(sum_ratio)))

            # input("?")

            # bkg_ts = 0.


            # exp = np.log(np.mean(sum_ratio) * len(local_data))
            # print("BKG TS:", bkg_ts, exp, np.mean(sum_ratio[mask]),
            #       np.mean(sum_ratio))
            # input("?")
            #
            # med_ts = np.sum((ratio_energy + ratio_spatial) * source_mc[
            #     "ow"])
            # print(med_ts)
            # input("/?")

            ts_vals.append(med_ts)
            # bkg_vals.append(bkg_ts)

        n_sigs = np.array(n_sigs)
        n_bkgs = np.array(n_bkgs)

        weights = weight_f(n_sigs, n_bkgs)
        # weights = (n_sigs / np.mean(n_sigs)) #/ np.sqrt(float(len(sources))) #*
        # np.median(n_bkgs)

        sum_n_sigs = np.sum(n_sigs * weights)
        sum_n_bkgs = np.sum(n_bkgs * weights)

        season_sig.append(sum_n_sigs)
        season_bkg.append(sum_n_bkgs)

        all_ns.append(n_sigs)
        all_nbkg.append(n_bkgs)

        ts_vals = np.array(ts_vals)

        # bkg_energy = llh.energy_weight_f(data)
        #
        # dists = [min([angular_distance(y["ra"], y["dec"], x, 0.0)
        #               for x in np.linspace(0, 2*np.pi, len(sources))])
        #          for y in data]
        #
        # bkg_spatial = ((1. / (
        #         2. * np.pi * data["sigma"] ** 2.) *
        #                      np.exp(-0.5 * ((dists/data["sigma"]) ** 2.)))
        #                     / (2 * np.pi))
        #
        # sum_ratio = np.array(bkg_energy * bkg_spatial)
        #
        # bkg_vals = np.sum(np.log(1. + sum_ratio / len(data)))

        bkg_vals = 0.

        # print(bkg_vals)
        # input("?")
        # print(ts_vals, ts_weight(n_sigs), ts_weight(n_sigs) * ts_vals)

        all_ts.append(ts_vals * ts_weight(n_sigs))
        all_bkg_ts.append(bkg_vals)# * ts_weight(n_sigs))


    season_sig = np.array(season_sig)
    season_bkg = np.array(season_bkg)

    season_weights = 1.

    int_sig = np.sum(season_sig * season_weights)
    int_bkg = np.sum(season_bkg * season_weights)

    disc_count = norm.ppf(norm.cdf(5.0), loc=int_bkg,
                          scale=np.sqrt(int_bkg))

    # all_ns = np.array(all_ns)
    # all_nbkg = np.array(all_nbkg)
    # weights = all_ns/max([max(x) for x in all_ns])
    #
    # print(all_ns, all_nbkg, weights)
    #
    # all_nbkg *= weights
    #
    # print(all_nbkg)
    # 
    # n_bkg = np.sum([np.sum(x) for x in all_nbkg])
    # n_s = np.sum(np.sum(x) for x in all_ns)
    #
    # print("disc:", disc_count)
    # print("Int sig:", int_sig)
    #
    # print("N bkg:", n_bkg)
    # print("Ns", n_s)
    # disc_count = norm.ppf(norm.cdf(5.0), loc=n_bkg, scale=np.sqrt(max(
    #     [max(x) for x in all_nbkg])))
    # int_sig = np.sum(np.sum(x) for x in all_ns*weights)
    # int_bkg = n_bkg
    # print("disc count:", disc_count)
    # print("Ns:", np.sum(np.sum(x) for x in all_ns*weights))

    # input("prompt?")

    all_ts = np.array(all_ts).T
    # print(all_ts, "for", n_s_tot)
    # print((n_tot - n_s_tot) * np.log1p(-n_s_tot / n_tot))
    # print("(Final)")
    # input("?")

    # print(all_ts, ts_weight(season_sig), all_ts*ts_weight(season_sig))
    # input("?")
    all_ts *= 2 * ts_weight(season_sig)

    sum_ts = np.sum(all_ts)

    all_bkg = np.array(all_bkg_ts).T
    all_bkg *= 2 * ts_weight(season_sig)
    sum_bkg = np.sum(all_bkg)
    # print("Sum bkg", sum_bkg)
    # input("?")
    # print("Sum_TS:", sum_ts)
    # print("Disc:", 25./sum_ts)

    scale = (25. + sum_bkg)/sum_ts

    scale *= (1. + 0.5 * np.log(len(sources)))

    # disc_pot = disc_count - int_bkg
    #
    # scale = disc_pot / int_sig
    #
    # # The approximate scaling for idealised + binned to unbinned
    # # Scales with high vs low statistics. For low statistics, you are
    # # effectively are background free, so don't take the 50% hit for only
    # # counting nearby neutrinos. In high-statics regime, previous study
    # # showed ~factor of 2 improvement for binned vs unbinned
    #
    # fudge_factor = (1.25 + 0.75 * np.tanh(np.log(sca)))
    # # fudge_factor = (1.25 + 0.75 * np.tanh(np.log(disc_count)))
    # fudge_factor = 2.0
    # fudge_factor *= 0.8
    # # fudge_factor *= 0.5
    #
    # scale /= fudge_factor

    # Convert from scale factor to flux units

    scale = k_to_flux(scale) * raw_scale

    print()
    print(
        "Estimated Discovery Potential is: {:.3g} GeV sr^-1 s^-1 cm^-2".format(
            scale
        ))
    print()
    # input("?")
    return scale

class AsimovEstimator(object):

    def __init__(self, seasons, inj_dict):

        self.seasons = seasons
        self.injectors = dict()

        for season in self.seasons.values():
            self.injectors[season.season_name] = season.make_injector(
                [], **inj_dict)

    def guess_discovery_potential(self, source_path):
        sources = load_catalogue(source_path)

        for inj in self.injectors.values():
            inj.update_sources(sources)

        return estimate_discovery_potential(self.injectors, sources)
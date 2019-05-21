from __future__ import division
from builtins import str
from builtins import range
import os
import numpy as np
import matplotlib.pyplot as plt
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.data.icecube.ps_tracks.ps_v003_p01 import IC86_234567_dict
from flarestack.shared import plot_output_dir
from flarestack.icecube_utils.dataset_loader import data_loader
from flarestack.core.astro import angular_distance
from numpy.lib.recfunctions import append_fields

basedir = plot_output_dir(
    "analyses/angular_error_floor/dynamic_pull_corrections/")

try:
    os.makedirs(basedir)
except OSError:
    pass

energy_bins = np.linspace(1., 10., 20 + 1)


# def get_data(season):
#     mc = data_loader(season["mc_path"], floor=False)
#     x = np.degrees(angular_distance(
#         mc["ra"], mc["dec"], mc["trueRa"], mc["trueDec"]))
#     y = np.degrees(mc["sigma"]) * 1.177
#     return mc, x, y

def weighted_quantile(values, quantiles, weight):
    """
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weight: array-like of the same length as `array`
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


sin_dec_bins = IC86_1_dict["sinDec bins"]


n_log_e_bins = 20 + 1
log_e_bins = np.linspace(2., 6., n_log_e_bins)
quantiles = np.linspace(0., 1., n_log_e_bins)

#mc_path = "/lustre/fs22/group/icecube/data_mirror/misc
# /CombinedTracks_AllSky_Uncorrected_MC.npy"

mc_path = IC86_234567_dict["mc_path"]
mc = data_loader(mc_path, floor=False)
exp = data_loader(IC86_234567_dict["exp_path"], floor=False)
# sin_dec_bins = weighted_quantile(
#     mc["sinDec"], quantiles, np.ones_like(mc["sinDec"]))

# for gamma in [1.0, 1.5, 2.0, 3.0, 3.7]:
for gamma in [2.0, 3.5]:
    # X = [sin_dec_bins for _ in range(n_log_e_bins)]
    # Y = []
    Z_quantile = np.ones((len(sin_dec_bins), n_log_e_bins))
    Z_uniform = np.ones((len(sin_dec_bins), n_log_e_bins))
    Z_floor = np.ones((len(sin_dec_bins), n_log_e_bins)) * np.nan
    Z_floor2 = np.ones((len(sin_dec_bins), n_log_e_bins)) * np.nan

    x = []
    y = []
    z = []

    subdir = basedir + str(gamma) + "/"

    try:
        os.makedirs(subdir)
    except OSError:
        pass

    for i, lower in enumerate(sin_dec_bins[:-1]):
        upper = sin_dec_bins[i + 1]
        mask = np.logical_and(
            mc["sinDec"] > lower,
            mc["sinDec"] < upper
        )
        cut_mc = mc[mask]

        percentile = np.ones_like(cut_mc["ra"]) * np.nan
        cut_mc = append_fields(
            cut_mc, 'percentile', percentile,
            usemask=False, dtypes=[np.float]
        )

        weights = cut_mc["ow"] * cut_mc["trueE"] ** - gamma
        # weights = np.ones_like(cut_mc["ow"])

        log_e_quantile_bins = weighted_quantile(
            cut_mc["logE"], quantiles, np.ones_like(cut_mc["ow"]))
        data_mask = np.logical_and(
            exp["sinDec"] > lower,
            exp["sinDec"] < upper
        )

        # log_e_quantile_bins = np.percentile(exp[data_mask]["logE"],
        #                                     quantiles)
        #
        # print log_e_quantile_bins
        # print log_e_quantile_bins

        # log_e_quantile_bins[-1] += 1e-5
        # Y.append(log_e_bins)

        # plt.figure()
        # plt.hist(cut_mc["logE"], log_e_quantile_bins, weights=weights, density=True)
        # plt.xlabel("log(Energy Proxy/GeV)")
        # plt.savefig(basedir + "hist.pdf")
        # plt.close()
        # raw_input("prompt")

        meds = []
        floors = []
        floor2s = []
        floor_corrected = []
        floor2_corrected = []
        ceilings = []
        x_vals = []

        for j, lower_e in enumerate(log_e_bins[:-1]):
            upper_e = log_e_bins[j + 1]
            e_mask = np.logical_and(
                cut_mc["logE"] >= lower_e,
                cut_mc["logE"] < upper_e
            )

            bin_mc = cut_mc[e_mask]

            x = np.degrees(angular_distance(
                bin_mc["ra"], bin_mc["dec"], bin_mc["trueRa"],
                bin_mc["trueDec"]))
            y = np.degrees(bin_mc["sigma"]) * 1.177

            if np.sum(e_mask) > 0:

                [floor, floor2, ceiling] = weighted_quantile(
                    x, [0.1, 0.25, 0.9], weights[e_mask])
                pull = x / y
                median_pull = weighted_quantile(
                    pull, 0.5, weights[e_mask])

                meds += [median_pull for _ in range(2)]
                floors += [floor for _ in range(2)]
                floor2s += [floor2 for _ in range(2)]
                floor_corrected += [floor * median_pull for _ in range(2)]
                floor2_corrected += [floor2 * median_pull for _ in range(2)]
                # ceilings += [ceiling for _ in range(2)]
                x_vals += [lower_e, upper_e]

                # convert_x = np.linspace(0.0, 1.0, 100)
                # convert_y = [weighted_quantile(bin_mc["sigma"], x, weights[
                #     e_mask]) for x in convert_x]
                # convert_y[0] = min(bin_mc["sigma"])
                # convert_y[-1] = max(bin_mc["sigma"])
                # # print convert_x
                # # print convert_y
                # # print len(bin_mc)
                # # raw_input("prompt")
                #
                # f = interp1d(convert_y, convert_x)
                #
                # cut_mc["percentile"][e_mask] = f(bin_mc["sigma"])
                #
                # mapmask = np.isnan(cut_mc["percentile"][e_mask])
                # # if np.sum(mapmask) > 0.:
                # print bin_mc["sigma"][mapmask], max(bin_mc["sigma"]),
                # print min(bin_mc["sigma"])
                # print cut_mc["percentile"]
                # raw_input("prompt")

                Z_uniform[i][j] = median_pull
                Z_floor[i][j] = floor * median_pull
                Z_floor2[i][j] = floor2 * median_pull
            else:
                Z_uniform[i][j] = np.nan

        for j, lower_e in enumerate(log_e_quantile_bins[:-1]):
            upper_e = log_e_quantile_bins[j + 1]
            e_mask = np.logical_and(
                cut_mc["logE"] >= lower_e,
                cut_mc["logE"] < upper_e
            )

            bin_mc = cut_mc[e_mask]

            x = np.degrees(angular_distance(
                bin_mc["ra"], bin_mc["dec"], bin_mc["trueRa"],
                bin_mc["trueDec"]))
            y = np.degrees(bin_mc["sigma"]) * 1.177

            [floor, ceiling] = weighted_quantile(
                x, [0.1, 0.9], weights[e_mask])
            pull = x / y
            median_pull = weighted_quantile(
                pull, 0.5, weights[e_mask])

            Z_quantile[i][j] = median_pull

        plt.figure()
        plt.plot(x_vals, meds, label="Median Pull")
        plt.plot(x_vals, floors, label="10% Quantile Uncorrected (deg)",
                 linestyle=":", color="orange")
        plt.plot(x_vals, floor2s, label="25% Quantile Uncorrected (deg)",
                 linestyle=":", color="green")
        plt.plot(x_vals, floor_corrected, label="10% Quantile (deg)",
                 color="orange")
        plt.plot(x_vals, floor2_corrected, label="25% Quantile (deg)",
                 color="green")
        # plt.plot(x_vals, med_errors, label="50% Quantile (deg)")
        plt.axhline(1.0, linestyle="--")
        # plt.plot(x_vals, ceilings, label="90% Quantile")
        plt.xlabel("log(Energy Proxy/GeV)")
        plt.legend()
        plt.savefig(subdir + "median_pulls_" + str(lower) + ".pdf")
        plt.close()

        # sigmas = np.linspace(0.0, 1.0, 10)
        #
        # med_qs = []
        # q_centers = 0.5 * (sigmas[1:] + sigmas[:-1])
        #
        # x = np.degrees(angular_distance(
        #     cut_mc["ra"], cut_mc["dec"], cut_mc["trueRa"],
        #     cut_mc["trueDec"]))
        # y = np.degrees(cut_mc["sigma"]) * 1.177
        # pulls = x/y
        #
        # base_pull = weighted_quantile(
        #     pulls, 0.5, weights)
        #
        # for k, lower_q in enumerate(sigmas[:-1]):
        #     upper_q = sigmas[k + 1]
        #
        #     q_mask = np.logical_and(
        #         cut_mc["percentile"] >= lower_q,
        #         cut_mc["percentile"] < upper_q
        #     )
        #
        #     bin_mc = cut_mc[q_mask]
        #
        #     x = np.degrees(angular_distance(
        #         bin_mc["ra"], bin_mc["dec"], bin_mc["trueRa"],
        #         bin_mc["trueDec"]))
        #     y = np.degrees(bin_mc["sigma"]) * 1.177
        #
        #     pull = x/y/base_pull
        #     median_pull = weighted_quantile(
        #         pull, 0.5, weights[q_mask])
        #
        #     med_qs.append(median_pull)
        #
        # plt.figure()
        # plt.plot(q_centers, med_qs, label="median pull")
        # plt.axhline(1.0, linestyle="--")
        # # plt.plot(x_vals, ceilings, label="90% Quantile")
        # plt.xlabel("Angular Error Percentile")
        # plt.ylabel("Pull / bin median pull")
        # plt.legend()
        # plt.savefig(subdir + "err_pulls_" + str(lower) + ".pdf")
        # plt.close()

    # sigmas = np.linspace(0.0, 1.0, 10)
    #
    # med_qs = []
    # q_centers = 0.5 * (sigmas[1:] + sigmas[:-1])
    #
    # x = np.degrees(angular_distance(
    #     mc["ra"], mc["dec"], mc["trueRa"],
    #     mc["trueDec"]))
    # y = np.degrees(mc["sigma"]) * 1.177
    # pulls = x / y
    #
    # weights = mc["ow"] * mc["trueE"] ** - gamma
    #
    # base_pull = weighted_quantile(
    #     pulls, 0.5, weights)
    #
    # for k, lower_q in enumerate(sigmas[:-1]):
    #     upper_q = sigmas[k + 1]
    #
    #     q_mask = np.logical_and(
    #         mc["percentile"] >= lower_q,
    #         mc["percentile"] < upper_q
    #     )
    #
    #     bin_mc = mc[q_mask]
    #
    #     x = np.degrees(angular_distance(
    #         bin_mc["ra"], bin_mc["dec"], bin_mc["trueRa"],
    #         bin_mc["trueDec"]))
    #     y = np.degrees(bin_mc["sigma"]) * 1.177
    #
    #     pull = x / y / base_pull
    #     median_pull = weighted_quantile(
    #         pull, 0.5, weights[q_mask])
    #
    #     med_qs.append(median_pull)
    #
    # plt.figure()
    # plt.plot(q_centers, med_qs, label="median pull")
    # plt.axhline(1.0, linestyle="--")
    # # plt.plot(x_vals, ceilings, label="90% Quantile")
    # plt.xlabel("Angular Error Percentile")
    # plt.ylabel("Pull / bin median pull")
    # plt.legend()
    # plt.savefig(subdir + "err_pulls_summed.pdf")
    # plt.close()

    for l, Z in enumerate([Z_floor, Z_floor2]):

        # Z = np.log(Z)

        plt.figure()
        ax = plt.subplot(111)
        X, Y = np.meshgrid(sin_dec_bins, log_e_bins)
        cbar = ax.pcolor(X, Y, Z.T, vmin=0.0, vmax=1.0, cmap="viridis",)
        plt.colorbar(cbar, label="Error (deg)")
        plt.xlabel(r"$\sin(\delta)$")
        plt.ylabel("Log(Energy proxy)")
        plt.savefig(basedir + "2D_meds_" + str(gamma) + "_" +
                    ["floor_10", "floor_25"][l] + ".pdf")
        plt.close()

    for l, Z in enumerate([Z_uniform, Z_quantile, ]):

        Z = np.log(Z)
        max_col = 1.0

        plt.figure()
        ax = plt.subplot(111)
        X, Y = np.meshgrid(sin_dec_bins,
                           [log_e_bins, quantiles, log_e_bins, log_e_bins][l])
        cbar = ax.pcolor(X, Y, Z.T, vmin=-max_col, vmax=max_col,
                         cmap="seismic",)
        plt.colorbar(cbar, label="Log(Pull)")
        plt.xlabel(r"$\sin(\delta)$")
        plt.ylabel(["Log(Energy proxy)",
                    "Unweighted energy proxy percentile",
                    "Log(Energy proxy)", "Log(Energy proxy)"][l])
        plt.savefig(basedir + "2D_meds_" + str(gamma) + "_" +
                    ["standard", "percentile"][l] + ".pdf")
        plt.close()




import numpy as np
import scipy.interpolate
import os
import numexpr
import resource
import matplotlib.pyplot as plt
from flarestack.data.icecube.gfu.gfu_v002_p01 import gfu_v002_p01
from flarestack.utils.dataset_loader import data_loader
from flarestack.core.energy_PDFs import PowerLaw
from flarestack.shared import plots_dir
import matplotlib.cm as cm

# season_dict = gfu_v002_p01[0]

# exp = data_loader(season_dict["exp_path"])
# mc = data_loader(season_dict["mc_path"])
#
# print mc.dtype.names
#
# print max(mc["time"]), min(mc["time"])

energy_pdf = PowerLaw()

azimuth_bins = np.linspace(0., 2*np.pi, 181)
sin_dec_bins = np.linspace(-1., 1., 11)

def create_2d_hist(sin_dec, az, sin_dec_bins, weights):
    """Creates a 2D histogram for a set of data (Experimental or Monte
    Carlo), in which the dataset is binned by sin(Declination) and
    Log(Energy). Weights the histogram by the values in the weights array.
    Normalises the histogram, such that the sum of each sin(Declination)
    column is equal to 1.

    :param sin_dec: Sin(Declination) array
    :param log_e: Log(Energy/GeV) array
    :param weights: Array of weights for event
    :return: Normalised histogram
    """
    # Produces the histogram
    hist_2d, binedges = np.histogramdd(
        (az, sin_dec), bins=(azimuth_bins, sin_dec_bins), weights=weights)
    n_dimensions = hist_2d.ndim

    # Normalises histogram
    norms = np.sum(hist_2d, axis=n_dimensions - 2)
    norms[norms == 0.] = 1.
    hist_2d /= norms

    return hist_2d

def create_2d_ratio_spline(exp, mc, sin_dec_bins, gamma):
    """Creates 2D histograms for both data and MC, in which the seasons
    are binned by Sin(Declination) and Log(Energy/GeV). Each histogram is
    normalised in Sin(Declination) bands. Then creates a histogram of the
    ratio of the Signal/Background histograms. In bins where there is
    simulation but no data, the ratio is set to the highest ratio
    value found for cases with both data and MC.  This is broadly
    unimportant for unblinded archival searches, because there will never
    be a situation in which a bin without any data will be queried. In all
    other cases, the ratio is set to 1.

    A 2D spline, of 2nd order in x and y, is then fit to the Log(Ratio),
    and returned.

    :return: 2D spline function
    """

    bkg_hist = create_2d_hist(exp["sinDec"], exp["azimuth"], sin_dec_bins,
                              weights=np.ones_like(exp["logE"]))

    sig_hist = create_2d_hist(np.sin(mc["trueDec"]), mc["trueAzimuth"],
                              sin_dec_bins,
                              weights=energy_pdf.weight_mc(mc, gamma))

    # Produces an array containing True if x > 0, False otherwise
    domain_bkg = bkg_hist > 0.
    domain_sig = sig_hist > 0.

    # Creates an array of ones as the default ratio
    ratio = np.ones_like(bkg_hist, dtype=np.float)
    # Bitwise Addition giving an Truth Array
    # Returns True if True in both Sig and Bkg, otherwise False
    mask = domain_bkg & domain_sig
    # Calculates the ratio sig/bkg for those entries True in Mask array
    ratio[mask] = (sig_hist[mask] / bkg_hist[mask])

    # Finds the maximum ratio
    max_ratio = np.amax(ratio)
    # Where true in sig and false in bkg, sets ratio to maximum ratio
    np.copyto(ratio, max_ratio, where=domain_sig & ~domain_bkg)

    # Sets bin centers, and order of spline (for x and y)
    sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.
    az_bin_center = (azimuth_bins[:-1] + azimuth_bins[1:]) / 2.
    order = 2

    # Fits a 2D spline function to the log of ratio array
    # This is 2nd order in both dimensions
    spline = scipy.interpolate.RectBivariateSpline(
        az_bin_center, sin_bin_center, np.log(ratio),
        kx=order, ky=order, s=0)

    return spline

def azimuth_proxy(data, season_dict):

    t = data[season_dict["MJD Time Key"]]

    sidereal_day = 364./365.

    res = t % sidereal_day

    az_offset = 2.54199002505 + 2.907

    az = az_offset + 2 * np.pi * res - data["ra"]
    #
    # az = (0.5 * np.pi

    while np.sum(az > 2 * np.pi) > 0:
        az[az > 2 * np.pi] -= 2 * np.pi

    while np.sum(az < 0) > 0:
        az[az < 0] += 2 * np.pi

    if "azimuth" in data.dtype.names:

        plt.figure()
        plt.hist(az - data["azimuth"], bins=180)

        # indices = np.linspace(1, len(az), 1000)
        #
        # mask = np.zeros(len(az), dtype=np.bool)
        #
        # for j in indices:
        #     mask[j - 1] = 1
        #
        # # print mask[:10]
        #
        # plt.scatter(az[mask], data["azimuth"][mask])

        plt.savefig(plots_dir + "azimuth/hist.pdf")
        plt.close()
        raw_input("prompt")

    # return data["azimuth"]

    return az

def plot_ratio(seasons):

    for season_dict in seasons:

        exp = data_loader(season_dict["exp_path"])
        mc = data_loader(season_dict["mc_path"])

        cut = 4.5

        exp_cut = exp[exp["logE"] > cut]
        mc_cut = mc[mc["logE"] > cut]

        print len(exp), len(exp_cut)
        print len(mc), len(mc_cut)

        print "Min energy", np.exp(min(exp["logE"]))

        print exp.dtype.names

        data = [
            ("exp", exp["sinDec"], azimuth_proxy(exp, season_dict)),
            ("mc", np.sin(mc["dec"]), azimuth_proxy(mc, season_dict)),
            (("exp_cut"), exp_cut["sinDec"],
             azimuth_proxy(exp_cut, season_dict)),
            (("mc cut"), np.sin(mc_cut["dec"]), azimuth_proxy(mc_cut,season_dict))

        ]

        root = plots_dir + "azimuth/" + season_dict["Data Sample"] + "/"

        for (name, sin_dec, az) in data:

            save_dir = root + name + "/"

            try:
                os.makedirs(save_dir)
            except OSError:
                pass

            if name in ["exp", "exp_cut"]:

                hist_2d = create_2d_hist(sin_dec, az,
                                         sin_dec_bins,
                                         weights=np.ones_like(sin_dec)).T

            else:
                ow = mc['ow']
                trueE = mc['trueE']
                weights = numexpr.evaluate('ow * trueE **(-2)')

                mem_use = str(
                    float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
                print ""
                print 'Memory usage max: %s (Gb)' % mem_use

                if name == "mc":

                    hist_2d = create_2d_hist(
                        sin_dec, az, sin_dec_bins,
                        weights=energy_pdf.weight_mc(mc, gamma=3.7)).T

                else:
                    hist_2d = create_2d_hist(
                        sin_dec, az, sin_dec_bins,
                        weights=energy_pdf.weight_mc(
                            mc_cut, gamma=3.7)).T


            hist_2d = np.array([x / np.mean(x) for x in hist_2d])

            plt.figure()
            ax = plt.subplot(111)

            sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.
            az_bin_center = (azimuth_bins[:-1] + azimuth_bins[1:]) / 2.

            X, Y = np.meshgrid(az_bin_center, sin_bin_center)
            cbar = ax.pcolormesh(X, Y, hist_2d, vmin=0.5, vmax=1.5,
                                 cmap=cm.get_cmap('seismic'))
            # ax.set_aspect('equal')

            plt.axis([min(azimuth_bins), max(azimuth_bins),
                      min(sin_dec_bins), max(sin_dec_bins),
                      ])
            ax.set_ylabel("Sin(Declination)")
            ax.set_xlabel("Azimuth")
            plt.colorbar(cbar)

            savename = save_dir + season_dict["Name"] + ".pdf"
            plt.savefig(savename)
            plt.close()

            spline = scipy.interpolate.RectBivariateSpline(
                az_bin_center, sin_bin_center, hist_2d.T,
                kx=3, ky=1, s=0)

            plt.figure()
            ax = plt.subplot(111)

            sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.
            az_bin_center = (azimuth_bins[:-1] + azimuth_bins[1:]) / 2.

            z = []

            for sin_dec in sin_dec_bins:
                z.append(list(spline(azimuth_bins, sin_dec).T.tolist()[0]))

            z = np.array(z).T

            X, Y = np.meshgrid(azimuth_bins, sin_dec_bins)
            cbar = ax.pcolormesh(X, Y, z.T, vmin=0.5, vmax=1.5,
                          cmap=cm.get_cmap('seismic'))
            # ax.set_aspect('equal')

            plt.axis([min(azimuth_bins), max(azimuth_bins),
                      min(sin_dec_bins), max(sin_dec_bins),
                      ])
            ax.set_ylabel("Sin(Declination)")
            ax.set_xlabel("Azimuth")
            plt.colorbar(cbar)

            savename = save_dir + season_dict["Name"] + "_spline.pdf"
            plt.savefig(savename)
            plt.close()

        upgoing = [azimuth_proxy(exp[exp["sinDec"] > 0.], season_dict),
                   azimuth_proxy(mc[mc["dec"] > 0.], season_dict)]

        downgoing = [azimuth_proxy(exp[exp["sinDec"] < 0.], season_dict),
                     azimuth_proxy(mc[mc["dec"] < 0.], season_dict)]

        upgoing_cut = [azimuth_proxy(exp_cut[exp_cut["sinDec"] > 0.],
                                     season_dict),
                   azimuth_proxy(mc_cut[mc_cut["dec"] > 0.], season_dict)]

        # downgoing_cut = [azimuth_proxy(exp_cut[exp_cut["sinDec"] < 0.],
        #                                season_dict),
        #              azimuth_proxy(mc_cut[mc_cut["dec"] < 0.], season_dict)]

        for i, dataset in enumerate([upgoing, downgoing, upgoing_cut]):
            plt.figure()
            ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

            weights = [np.ones_like(x) / float(len(x)) for x in dataset]

            n, edges, patches = plt.hist(
                dataset, bins=azimuth_bins, histtype='step',
                weights=weights, label=["Data", "MC"], color=["red", "b"])

            med_bkg = np.median(n[0])

            mask = n[0] > med_bkg
            over = n[0][mask]

            sum_over = np.sum(over - med_bkg)

            title = season_dict["Name"] + " " + [
                "upgoing", "downgoing", "upgoing [Log(E) > " + str(cut) + "]",
                "downgoing [Log(E) > " + str(cut) + "]"][i]

            plt.axhline(med_bkg, linestyle="--", color="k", linewidth=0.5,)

            message = "{0:.2f}".format(sum_over)
            plt.annotate(message, xy=(0.05, 0.9),
                         xycoords="axes fraction", color="red")

            mids = (edges[:-1] + edges[1:]) / 2.

            fills = np.array(n[0])
            fills[~mask] = med_bkg

            ax1.fill_between(mids, fills, med_bkg, facecolor='red',
                             alpha=0.5)

            ax1.set_ylim(ymin=0.0)

            plt.title(title)
            plt.legend()
            plt.xlabel("Azimuth")
            plt.ylabel("Fraction of Total")
            ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1,
                                   sharex=ax1)

            plt.plot(mids, n[1]/n[0], color="orange")
            plt.axhline(1., linestyle="--", color="k")
            plt.ylabel("Ratio (MC/Data)")
            plt.xlabel("Azimuth (rad)")
            xticklabels = ax1.get_xticklabels()
            plt.setp(xticklabels, visible=False)
            yticklabels = ax1.get_yticklabels()
            plt.setp(yticklabels[0], visible=False)
            plt.subplots_adjust(hspace=0.001)
            # ax2.set_xlim(0.0, 2*np.pi)

            plt.savefig(root + title + " 1D.pdf")
            plt.close()

# plot_ratio(txs_sample_v1)
plot_ratio(gfu_v002_p01)
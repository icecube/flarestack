import numpy as np
import os
import scipy.interpolate
import cPickle as Pickle
from flarestack.shared import gamma_precision, SoB_spline_path, \
    bkg_spline_path, dataset_plot_dir
from flarestack.core.energy_PDFs import PowerLaw
from flarestack.utils.dataset_loader import data_loader
import matplotlib.pyplot as plt


# sin_dec_bins = np.unique(np.concatenate([
#             np.linspace(-1., -0.9, 2 + 1),
#             np.linspace(-0.9, -0.2, 8 + 1),
#             np.linspace(-0.2, 0.2, 15 + 1),
#             np.linspace(0.2, 0.9, 12 + 1),
#             np.linspace(0.9, 1., 2 + 1),
#         ]))

energy_bins = np.linspace(1., 10., 50 + 1)

energy_pdf = PowerLaw()

def _around(value):
    """Produces an array in which the precision of the value
    is rounded to the nearest integer. This is then multiplied
    by the precision, and the new value is returned.

    :param value: value to be processed
    :return: value after processed
    """
    return np.around(float(value) / gamma_precision) * gamma_precision


gamma_points = np.arange(0.7, 4.3, gamma_precision)
gamma_support_points = set([_around(i) for i in gamma_points])


def create_2d_hist(sin_dec, log_e, sin_dec_bins, weights):
    """Creates a 2D histogram for a set of data (Experimental or Monte
    Carlo), in which the dataset is binned by sin(Declination) and
    Log(Energy). Weights the histogram by the values in the weights array.
    Normalises the histogram, such that the sum of each sin(Declination)
    column is equal to 1.

    :param sin_dec: Sin(Declination) array
    :param log_e: Log(Energy/GeV) array
    :param sin_dec_bins: Bins of Sin(Declination to be used)
    :param weights: Array of weights for event
    :return: Normalised histogram
    """
    # Produces the histogram
    hist_2d, binedges = np.histogramdd(
        (log_e, sin_dec), bins=(energy_bins, sin_dec_bins), weights=weights)
    # n_dimensions = hist_2d.ndim

    # # Normalises histogram
    # norms = np.sum(hist_2d, axis=n_dimensions - 2)
    # norms[norms == 0.] = 1.
    # hist_2d /= norms

    return hist_2d

# def create_2d_gamma_energy(sin_dec, log_e, sin_dec_bins, weights):
#     """Creates a 2D histogram for a set of data (Experimental or Monte
#     Carlo), in which the dataset is binned by sin(Declination) and
#     Log(Energy). Weights the histogram by the values in the weights array.
#     Normalises the histogram, such that the sum of each sin(Declination)
#     column is equal to 1.
#
#     :param sin_dec: Sin(Declination) array
#     :param log_e: Log(Energy/GeV) array
#     :param sin_dec_bins: Bins of Sin(Declination to be used)
#     :param weights: Array of weights for event
#     :return: Normalised histogram
#     """
#     # Produces the histogram
#     hist_2d, binedges = np.histogramdd(
#         (log_e, sin_dec), bins=(energy_bins, sin_dec_bins), weights=weights)
#     n_dimensions = hist_2d.ndim
#
#     # Normalises histogram
#     norms = np.sum(hist_2d, axis=n_dimensions - 2)
#     norms[norms == 0.] = 1.
#     hist_2d /= norms
#
#     return hist_2d


def create_bkg_2d_hist(exp, sin_dec_bins):
    """Creates a background 2D logE/sinDec histogram.

    :param exp: Experimental data
    :param sin_dec_bins: Bins of Sin(Declination to be used)
    :return: 2D histogram
    """
    return create_2d_hist(exp["sinDec"], exp["logE"], sin_dec_bins,
                          weights=np.ones_like(exp["logE"]))


def create_sig_2d_hist(mc, sin_dec_bins, gamma):
    """Creates a signal 2D logE/sinDec histogram.

    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :param gamma: Spectral Index
    :return: 2D histogram
    """
    return create_2d_hist(mc["sinDec"], mc["logE"], sin_dec_bins,
                          weights=energy_pdf.weight_mc(mc, gamma))


def create_2d_ratio_hist(exp, mc, sin_dec_bins, gamma):
    """Creates a 2D histogram for both data and MC, in which the seasons
    are binned by Sin(Declination) and Log(Energy/GeV). Each histogram is
    normalised in Sin(Declination) bands. Then creates a histogram of the
    ratio of the Signal/Background histograms. In bins where there is
    simulation but no data, a count of 1 is assigned to the background
    histogram.  This is broadly unimportant for unblinding archival searches,
    because there will never be a situation in which a bin without any data
    will be queried. In all other cases, the ratio is set to 1.

    :param exp: Experimental data
    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :param gamma: Spectral Index
    :return: ratio histogram
    """

    bkg_hist = create_bkg_2d_hist(exp, sin_dec_bins)
    sig_hist = create_sig_2d_hist(mc, sin_dec_bins, gamma)
    n_dimensions = sig_hist.ndim
    norms = np.sum(sig_hist, axis=n_dimensions - 2)
    norms[norms == 0.] = 1.
    sig_hist /= norms

    # # Produces an array containing True if x > 0, False otherwise
    # domain_bkg = bkg_hist > 0.
    # domain_sig = sig_hist > 0.
    # Creates an array of ones as the default ratio
    ratio = np.ones_like(bkg_hist, dtype=np.float)

    for i, bkg_row in enumerate(bkg_hist.T):
        sig_row = sig_hist.T[i]

        # int_row = np.array([
        #     np.sum(bkg_row[:j])/np.sum(bkg_row) for j, _ in enumerate(bkg_row)
        # ])

        fill_mask = (bkg_row == 0.) & (sig_row > 0.)
        bkg_row[fill_mask] = 1.
        # bkg_row /= np.sum(bkg_row)

        mask = (bkg_row > 0.) & (sig_row > 0.)
        r = np.ones_like(bkg_row)
        r[mask] = sig_row[mask] / bkg_row[mask] * np.sum(bkg_row)

        # s_mask = (int_row < 0.5)
        #
        # if np.mean(r[s_mask]) > 1:
        #     max_mask = fill_mask & (sig_row > 0.) & s_mask
        # else:
        #     max_mask = fill_mask & (sig_row > 0.) & ~s_mask
        #
        # r[max_mask] = max(r)

        # Fill all bins with MC and no data, which have below-median energy,
        # with 1

        # mask = (bkg_row == 0.) & (sig_row > 0.) & (int_row < 0.5)

        ratio.T[i] = r

        # s_mask = (int_row > 0.5)
        #
        #
        # print "Above", np.sum(r[s_mask]), np.mean(),
        # print "below", np.sum(r[~s_mask]), np.mean(r[~s_mask])
        #
        # raw_input("prompt")
    # Creates an array of ones as the default ratio
    # ratio = np.ones_like(bkg_hist, dtype=np.float)
    # Bitwise Addition giving an Truth Array
    # Returns True if True in both Sig and Bkg, otherwise False
    # mask = domain_bkg & domain_sig
    # mask = (sig_hist > 0.) & (bkg_hist > 0.)
    # Calculates the ratio sig/bkg for those entries True in Mask array
    # # ratio[mask] = (sig_hist[mask] / bkg_hist[mask])
    #
    # for i, ratio_row in enumerate(ratio.T):
    #     bkg_row = bkg_hist.T[i]
    #     mc_row = sig_hist.T[i]
    #
    #     # Fill events with MC and no data, but with above-median energy,
    #     # with the maximum S/B value
    #
    #     mask = (bkg_row == 0.) & (mc_row > 0.)
    #     ratio_row[mask] = max(ratio_row)

    return ratio


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

    :param exp: Experimental data
    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :param gamma: Spectral Index
    :return: 2D spline function
    """

    ratio = create_2d_ratio_hist(exp, mc, sin_dec_bins, gamma)

    # Sets bin centers, and order of spline (for x and y)
    sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.
    log_e_bin_center = (energy_bins[:-1] + energy_bins[1:]) / 2.
    order = 2

    # Fits a 2D spline function to the log of ratio array
    # This is 2nd order in both dimensions
    spline = scipy.interpolate.RectBivariateSpline(
        log_e_bin_center, sin_bin_center, np.log(ratio),
        kx=order, ky=order, s=0)

    return spline


def create_2d_splines(exp, mc, sin_dec_bins):
    """If gamma will not be fit, then calculates the Log(Signal/Background)
    2D PDF for the fixed value self.default_gamma. Fits a spline to each
    histogram, and saves the spline in a dictionary.

    If gamma should be fit, instead loops over each value of gamma in
    self.gamma_support_points. For each gamma value, the spline creation
    is repeated, and saved as a dictionary entry.

    In either case, returns the dictionary of spline/splines.

    :param exp: Experimental data
    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :return: Dictionary of 2D Log(Signal/Background) splines
    """
    splines = dict()

    for gamma in gamma_support_points:
        splines[gamma] = create_2d_ratio_spline(exp, mc, sin_dec_bins, gamma)

    return splines


def create_bkg_spatial_spline(exp, sin_dec_bins):
    """Creates the spatial PDF for background.
    Generates a histogram for the exp. distribution in sin declination.
    Fits a spline function to the distribution, giving a spatial PDF.
    Returns this spatial PDF.

    :param exp: Experimental data (background)
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :return: Background spline function
    """
    sin_dec_range = (np.min(sin_dec_bins), np.max(sin_dec_bins))
    hist, bins = np.histogram(
        exp['sinDec'], density=True, bins=sin_dec_bins, range=sin_dec_range)

    bins = np.concatenate([bins[:1], bins, bins[-1:]])
    hist = np.concatenate([hist[:1], hist, hist[-1:]])

    bkg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
                            (bins[1:] + bins[:-1]) / 2.,
                            np.log(hist), k=2)
    return bkg_spline


def make_spline(seasons):

    print "Splines will be made to calculate the Signal/Background ratio of " \
          "the MC to data. The MC will be weighted with a power law, for each" \
          " gamma in:"
    print list(gamma_support_points)

    for season in seasons:
        try:
            print "Making splines for", season["Name"]
            path = SoB_spline_path(season)

            bkg_path = bkg_spline_path(season)

            exp = data_loader(season["exp_path"])
            mc = data_loader(season["mc_path"])

            sin_dec_bins = season["sinDec bins"]

            splines = create_2d_splines(exp, mc, sin_dec_bins)

            print "Saving to", path

            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass

            with open(path, "wb") as f:
                Pickle.dump(splines, f)

            base_plot_path = dataset_plot_dir + "Signal_over_background/" +\
                             season["Data Sample"] + "/" + season["Name"] + "/"

            def make_plot(hist, savepath, normed=True):
                if normed:
                    norms = np.sum(hist, axis=hist.ndim - 2)
                    norms[norms == 0.] = 1.
                    hist /= norms
                else:
                    pass
                    hist = np.log(np.array(hist))
                plt.figure()
                ax = plt.subplot(111)
                X, Y = np.meshgrid(sin_dec_bins, energy_bins)
                if not normed:
                    max_col = min(abs(min([min(row) for row in hist.T])),
                                  max([max(row) for row in hist.T]))
                    cbar = ax.pcolormesh(X, Y, hist, cmap="seismic",
                                         vmin=-5, vmax=5)
                    plt.colorbar(cbar, label="Log(Signal/Background)")
                else:
                    hist[hist == 0.] = np.nan
                    cbar = ax.pcolormesh(X, Y, hist)
                    plt.colorbar(cbar, label="Column-normalised density")
                plt.xlabel(r"$\sin(\delta)$")
                plt.ylabel("log(Energy)")
                plt.savefig(savepath)
                plt.close()

            exp_hist = create_bkg_2d_hist(exp, sin_dec_bins)

            # Generate plots
            for gamma in np.linspace(1.0, 4.0, 7):

                plot_path = base_plot_path + "gamma=" + str(gamma) + "/"

                try:
                    os.makedirs(plot_path)
                except OSError:
                    pass

                mc_hist = create_sig_2d_hist(mc, sin_dec_bins, gamma)

                make_plot(mc_hist, savepath=plot_path + "sig.pdf")
                make_plot(create_2d_ratio_hist(exp, mc, sin_dec_bins, gamma),
                          savepath=plot_path + "SoB.pdf", normed=False)

                Z = []
                for s in sin_dec_bins:
                    z_line = []
                    for e in energy_bins:
                        z_line.append(splines[gamma](e, s)[0][0])
                    Z.append(z_line)

                Z = np.array(Z).T

                max_col = min(abs(min([min(row) for row in Z])),
                              max([max(row) for row in Z]))

                plt.figure()
                ax = plt.subplot(111)
                X, Y = np.meshgrid(sin_dec_bins, energy_bins)
                cbar = ax.pcolormesh(X, Y, Z, cmap="seismic",
                                     vmin=-max_col, vmax=max_col)
                plt.colorbar(cbar, label="Log(Signal/Background)")
                plt.xlabel(r"$\sin(\delta)$")
                plt.ylabel("log(Energy)")
                plt.savefig(plot_path + "spline.pdf")
                plt.close()

            make_plot(exp_hist,
                      savepath=base_plot_path + "bkg.pdf")

            # raw_input("prompt")

            bkg_spline = create_bkg_spatial_spline(exp, sin_dec_bins)

            print "Saving to", bkg_path

            try:
                os.makedirs(os.path.dirname(bkg_path))
            except OSError:
                pass

            with open(bkg_path, "wb") as f:
                Pickle.dump(bkg_spline, f)

        except IOError:
            pass


def load_spline(season):
    path = SoB_spline_path(season)

    print "Loading from", path

    with open(path) as f:
        res = Pickle.load(f)

    return res


def load_bkg_spatial_spline(season):
    path = bkg_spline_path(season)

    with open(path) as f:
        res = Pickle.load(f)

    return res
#
# from data.icecube_pointsource_7_year import ps_7year
# from data.icecube_gfu_2point5_year import gfu_v002_p01
# make_spline(gfu_v002_p01)
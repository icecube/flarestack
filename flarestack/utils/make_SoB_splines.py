import logging
import numpy as np
import os
import scipy.interpolate
import pickle as Pickle
from flarestack.shared import gamma_precision, SoB_spline_path, \
    bkg_spline_path, dataset_plot_dir, get_base_sob_plot_dir
from flarestack.core.energy_pdf import PowerLaw
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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


def create_2d_hist(sin_dec, log_e, sin_dec_bins, log_e_bins, weights):
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
        (log_e, sin_dec), bins=(log_e_bins, sin_dec_bins), weights=weights)\

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


def create_bkg_2d_hist(exp, sin_dec_bins, log_e_bins):
    """Creates a background 2D logE/sinDec histogram.

    :param exp: Experimental data
    :param sin_dec_bins: Bins of Sin(Declination to be used)
    :return: 2D histogram
    """
    return create_2d_hist(exp["sinDec"], exp["logE"], sin_dec_bins, log_e_bins,
                          weights=exp["weight"])


def create_sig_2d_hist(mc, sin_dec_bins, log_e_bins, weight_function):
    """Creates a signal 2D logE/sinDec histogram.

    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :param weight_function: Weight Function
    :return: 2D histogram
    """

    return create_2d_hist(mc["sinDec"], mc["logE"], sin_dec_bins, log_e_bins,
                          weights=weight_function(mc))


def create_2d_ratio_hist(exp, mc, sin_dec_bins, log_e_bins, weight_function):
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
    :param weight_function: Weight Function
    :return: ratio histogram
    """

    bkg_hist = create_bkg_2d_hist(exp, sin_dec_bins, log_e_bins)
    sig_hist = create_sig_2d_hist(mc, sin_dec_bins, log_e_bins, weight_function)
    n_dimensions = sig_hist.ndim
    norms = np.sum(sig_hist, axis=n_dimensions - 2)
    norms[norms == 0.] = 1.
    sig_hist /= norms

    ratio = np.ones_like(bkg_hist, dtype=np.float)

    for i, bkg_row in enumerate(bkg_hist.T):
        sig_row = sig_hist.T[i]

        fill_mask = (bkg_row == 0.) & (sig_row > 0.)
        bkg_row[fill_mask] = 1.
        # bkg_row /= np.sum(bkg_row)

        mask = (bkg_row > 0.) & (sig_row > 0.)
        r = np.ones_like(bkg_row)
        r[mask] = sig_row[mask] / (bkg_row[mask] / np.sum(bkg_row))

        ratio.T[i] = r
    #     print(max(sig_row), np.median(bkg_row), np.sum(bkg_row))
    #     print(r)
    #     input("prompt")
    #
    # input("prompt")

    return ratio


def create_2d_ratio_spline(exp, mc, sin_dec_bins, log_e_bins, weight_function):
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
    :param weight_function: Weight Function
    :return: 2D spline function
    """

    ratio = create_2d_ratio_hist(exp, mc, sin_dec_bins, log_e_bins,
                                 weight_function)

    spline = make_2d_spline_from_hist(ratio, sin_dec_bins, log_e_bins)

    return spline


def make_2d_spline_from_hist(ratio, sin_dec_bins, log_e_bins):

    # Sets bin centers, and order of spline (for x and y)
    sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.
    log_e_bin_center = (log_e_bins[:-1] + log_e_bins[1:]) / 2.
    order = 1

    # Fits a 2D spline function to the log of ratio array
    # This is 2nd order in both dimensions
    spline = scipy.interpolate.RectBivariateSpline(
        log_e_bin_center, sin_bin_center, np.log(ratio),
        kx=order, ky=order, s=0)
    return spline


def create_gamma_2d_ratio_spline(exp, mc, sin_dec_bins, log_e_bins, gamma):
    """Creates a 2D gamma ratio spline by creating a function that weights MC
    assuming a power law of spectral index gamma.

    :param exp: Experimental data
    :param mc: MC Simulations
    :param sin_dec_bins: Bins of Sin(Declination) to be used
    :param gamma: Spectral Index
    :return: 2D spline function
    """

    def weight_function(sig_mc):
        return energy_pdf.weight_mc(sig_mc, gamma)

    return create_2d_ratio_spline(exp, mc, sin_dec_bins, log_e_bins,
                                  weight_function)


def create_2d_splines(exp, mc, sin_dec_bins, log_e_bins):
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
        splines[gamma] = create_gamma_2d_ratio_spline(
            exp, mc, sin_dec_bins, log_e_bins, gamma)

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
        exp['sinDec'], density=True, bins=sin_dec_bins, range=sin_dec_range,
        weights=exp["weight"]
    )

    bins = np.concatenate([bins[:1], bins, bins[-1:]])
    hist = np.concatenate([hist[:1], hist, hist[-1:]])

    bkg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
                            (bins[1:] + bins[:-1]) / 2.,
                            np.log(hist), k=2)
    return bkg_spline


def make_spline(seasons):

    logger.info("Splines will be made to calculate the Signal/Background ratio of " \
          "the MC to data. The MC will be weighted with a power law, for each" \
          " gamma in: {0}".format(list(gamma_support_points)))

    for season in seasons.values():
        SoB_path = SoB_spline_path(season)
        make_individual_spline_set(season, SoB_path)
        make_background_spline(season)


def make_plot(hist, savepath, x_bins, y_bins, normed=True, log_min=5,
              label_x=r"$\sin(\delta)$", label_y="log(Energy)"):
    if normed:
        norms = np.sum(hist, axis=hist.ndim - 2)
        norms[norms == 0.] = 1.
        hist /= norms
    else:
        hist = np.log(np.array(hist))
    plt.figure()
    ax = plt.subplot(111)
    X, Y = np.meshgrid(x_bins, y_bins)
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
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.savefig(savepath)
    plt.close()

def make_individual_spline_set(season, SoB_path):
    try:
        logger.info("Making splines for {0}".format(season.season_name))
        # path = SoB_spline_path(season)

        exp = season.get_background_model()
        mc = season.get_pseudo_mc()

        sin_dec_bins = season.sin_dec_bins
        log_e_bins = season.log_e_bins

        splines = create_2d_splines(exp, mc, sin_dec_bins, log_e_bins)

        logger.info("Saving to {0}".format(SoB_path))

        try:
            os.makedirs(os.path.dirname(SoB_path))
        except OSError:
            pass

        with open(SoB_path, "wb") as f:
            Pickle.dump(splines, f)

        base_plot_path = get_base_sob_plot_dir(season)

        exp_hist = create_bkg_2d_hist(exp, sin_dec_bins, log_e_bins)

        # Generate plots
        for gamma in np.linspace(1.0, 4.0, 7):

            plot_path = base_plot_path + "gamma=" + str(gamma) + "/"

            try:
                os.makedirs(plot_path)
            except OSError:
                pass

            def weight_function(sig_mc):
                return energy_pdf.weight_mc(sig_mc, gamma)

            mc_hist = create_sig_2d_hist(mc, sin_dec_bins, log_e_bins,
                                         weight_function)

            make_plot(mc_hist, plot_path + "sig.pdf", sin_dec_bins, log_e_bins)
            make_plot(create_2d_ratio_hist(exp, mc, sin_dec_bins, log_e_bins,
                                           weight_function),
                      plot_path + "SoB.pdf", sin_dec_bins, log_e_bins, normed=False)

            Z = []
            for s in sin_dec_bins:
                z_line = []
                for e in log_e_bins:
                    z_line.append(splines[gamma](e, s)[0][0])
                Z.append(z_line)

            Z = np.array(Z).T

            max_col = min(abs(min([min(row) for row in Z])),
                          max([max(row) for row in Z]))

            plt.figure()
            ax = plt.subplot(111)
            X, Y = np.meshgrid(sin_dec_bins, log_e_bins)
            cbar = ax.pcolormesh(X, Y, Z, cmap="seismic",
                                 vmin=-max_col, vmax=max_col)
            plt.colorbar(cbar, label="Log(Signal/Background)")
            plt.xlabel(r"$\sin(\delta)$")
            plt.ylabel("log(Energy)")
            plt.savefig(plot_path + "spline.pdf")
            plt.close()

        make_plot(exp_hist,
                  savepath=base_plot_path + "bkg.pdf", x_bins=sin_dec_bins, y_bins=log_e_bins)

        del mc

    except IOError:
        pass


def make_background_spline(season):
    bkg_path = bkg_spline_path(season)
    bkg = season.get_background_model()
    sin_dec_bins = season.sin_dec_bins

    bkg_spline = create_bkg_spatial_spline(bkg, sin_dec_bins)

    logger.info("Saving to".format(bkg_path))
    try:
        os.makedirs(os.path.dirname(bkg_path))
    except OSError:
        pass

    with open(bkg_path, "wb") as f:
        Pickle.dump(bkg_spline, f)

    x_range = np.linspace(sin_dec_bins[0], sin_dec_bins[-1], 101)
    plt.figure()
    plt.plot(x_range, np.exp(bkg_spline(x_range)))
    plt.ylabel(r"$P_{bkg}$ (spatial)")
    plt.xlabel(r"$\sin(\delta)$")
    savepath = get_base_sob_plot_dir(season)

    try:
        os.makedirs(os.path.dirname(savepath))
    except OSError:
        pass

    plt.savefig(savepath + "bkg_spatial.pdf")
    plt.close()


def load_spline(season):
    path = SoB_spline_path(season)

    logger.debug("Loading from {0}".format(path))

    with open(path, "rb") as f:
        res = Pickle.load(f)

    return res


def load_bkg_spatial_spline(season):
    path = bkg_spline_path(season)

    logger.debug("Loading from {0}".format(path))

    with open(path, "rb") as f:
        res = Pickle.load(f)

    return res
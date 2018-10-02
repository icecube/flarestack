import numpy as np
import os
import scipy.interpolate
import cPickle as Pickle
from flarestack.shared import gamma_precision, SoB_spline_path, bkg_spline_path
from flarestack.core.energy_PDFs import PowerLaw
from flarestack.utils.dataset_loader import data_loader


# sin_dec_bins = np.unique(np.concatenate([
#             np.linspace(-1., -0.9, 2 + 1),
#             np.linspace(-0.9, -0.2, 8 + 1),
#             np.linspace(-0.2, 0.2, 15 + 1),
#             np.linspace(0.2, 0.9, 12 + 1),
#             np.linspace(0.9, 1., 2 + 1),
#         ]))

energy_bins = np.linspace(1., 10., 40 + 1)

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
    :param weights: Array of weights for event
    :return: Normalised histogram
    """
    # Produces the histogram
    hist_2d, binedges = np.histogramdd(
        (log_e, sin_dec), bins=(energy_bins, sin_dec_bins), weights=weights)
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

    bkg_hist = create_2d_hist(exp["sinDec"], exp["logE"], sin_dec_bins,
                              weights=np.ones_like(exp["logE"]))

    sig_hist = create_2d_hist(np.sin(mc["trueDec"]), mc["logE"], sin_dec_bins,
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
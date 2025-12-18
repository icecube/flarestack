import logging
import os
import pickle as Pickle
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from flarestack.core.energy_pdf import PowerLaw
from flarestack.shared import (
    SoB_spline_path,
    bkg_spline_path,
    default_gamma_precision,
    default_smoothing_order,
    flarestack_gamma_precision,
    get_base_sob_plot_dir,
)

logger = logging.getLogger(__name__)
energy_pdf = PowerLaw()


def get_gamma_precision(precision=flarestack_gamma_precision):
    """Returns the precision in gamma that is used.

    :param precision: Specify which precision to use. Default is 0.025 for 'flarestack'.
    Can also provide default of llh codes by name, either 'skylab' or 'flarestack'.
    :return: Precision as a float
    """
    if isinstance(precision, float):
        return precision

    elif isinstance(precision, str):
        if precision in default_gamma_precision:
            return default_gamma_precision[precision]
        else:
            raise ValueError(
                f"No gamma precision value defined for {precision}!"
                f"Choose: {default_gamma_precision.keys()}"
            )

    else:
        raise TypeError(
            f"Type {type(precision)} of {precision} not understood for variable gamma precision"
        )


def _around(value, precision=flarestack_gamma_precision):
    """Produces an array in which the precision of the value
    is rounded to the nearest integer. This is then multiplied
    by the precision, and the new value is returned.

    :param precision: Specify which precision to use. Default to the standard precision.
    Can also provide default of llh codes by name, either 'skylab' or 'flarestack'.
    :return: value after processed
    """
    return np.around(
        float(value) / get_gamma_precision(precision)
    ) * get_gamma_precision(precision)


def get_gamma_support_points(precision=flarestack_gamma_precision):
    """Return the gamma support points based on the gamma precision

    :param precision: Specify which precision to use.
    Can also provide default of llh codes by name, either 'skylab' or 'flarestack'.
    Default is 0.025 for 'flarestack'.
    :return: Gamma support points
    """
    gamma_points = np.arange(0.7, 4.3, get_gamma_precision(precision=precision))
    return set([_around(i, precision=precision) for i in gamma_points])


class SoB_splines:
    """Base class for Signal over Background splines
    which are used as the energy pdf in the llh.
    The splines are created per season from the
    2D histograms of the S/B ratio in sin(DEC) and Log(Energy/GeV).
    The background and signal histograms are created separately,
    weighted according to the chosen background flux model and
    the power-law injected signal respectively.
    The ratio of these 2D histograms is used to make the spline.
    """

    subclasses: dict[str, type["SoB_splines"]] = {}

    def __init__(self, season, SoB_dict) -> None:
        self.season = season
        self.sob_dict = SoB_dict
        self.spline_name = SoB_dict["bkg_model_name"]

        self.smoothing_order = SoB_dict["smoothing_order"]
        if isinstance(self.smoothing_order, str):
            if self.smoothing_order in default_smoothing_order.keys():
                self.smoothing_order = default_smoothing_order[self.smoothing_order]
            else:
                raise ValueError(
                    f"Smoothing order for {self.smoothing_order} not known!"
                )

        self.gamma_precision = SoB_dict["gamma_precision"]

    @classmethod
    def register_subclass(cls, spline_name):
        """Adds a new subclass of SoB_splines, with class name equal to
        "bkg_model_name".
        """

        def decorator(subclass):
            cls.subclasses[spline_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, season, SoB_dict) -> "SoB_splines":
        try:
            spline_name = SoB_dict["bkg_model_name"]
        except KeyError:
            spline_name = "no_difffuse"
            logger.info("No 'bkg_model_name' in sob dict, default to atmospheric-only")

        if spline_name not in cls.subclasses:
            raise ValueError("Bad SoB spline name {}".format(spline_name))

        return cls.subclasses[spline_name](season, SoB_dict)

    def bkg_weights(self, exp: Table) -> np.ndarray:
        raise NotImplementedError

    # ==============================================================================
    # BACKGROUND SPATIAL PDF
    # ==============================================================================

    def create_bkg_spatial_spline(self, exp, sin_dec_bins):
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
            exp["sinDec"],
            density=True,
            bins=sin_dec_bins,
            range=sin_dec_range,
            weights=self.bkg_weights(exp),
        )

        bins = np.concatenate([bins[:1], bins, bins[-1:]])
        hist = np.concatenate([hist[:1], hist, hist[-1:]])

        bkg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            (bins[1:] + bins[:-1]) / 2.0, np.log(hist), k=2
        )
        return bkg_spline

    def make_background_spline(self, season):
        bkg_path = bkg_spline_path(season, self.spline_name)
        bkg = season.get_background_model()
        sin_dec_bins = season.sin_dec_bins

        bkg_spline = self.create_bkg_spatial_spline(bkg, sin_dec_bins)

        logger.info(f"Saving bakcground spatial spline to {bkg_path}")
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

        sfx = f"{self.spline_name}_" if self.spline_name != "no_difffuse" else ""
        plt.savefig(savepath + sfx + "bkg_spatial.pdf")
        plt.close()

    def load_bkg_spatial_spline(self, season):
        path = bkg_spline_path(season, self.spline_name)

        logger.debug(f"Loading background spatial spline from {path}")

        try:
            with open(path, "rb") as f:
                res = Pickle.load(f)
        except FileNotFoundError as err:
            logger.info(
                f"No cached spline found at {path}. Creating this file instead."
            )
            logger.info(f"Cause: {err}")
            self.make_background_spline(season)
            with open(path, "rb") as f:
                res = Pickle.load(f)

        except ModuleNotFoundError as err:
            logger.error(
                "A spline was found but it seems incompatible with this installation."
            )
            logger.error(f"Cause: {err}")
            raise

        return res

    # ==============================================================================
    # ENERGY PDF SPLINES
    # ==============================================================================

    @staticmethod
    def create_2d_hist(sin_dec, log_e, sin_dec_bins, log_e_bins, weights):
        """Creates a 2D histogram for a set of data (Experimental or Monte
        Carlo), in which the dataset is binned by sin(Declination) and
        Log(Energy). Weights the histogram by the values in the weights array.
        Normalises the histogram, such that the sum of each sin(Declination)
        column is equal to 1.

        :param sin_dec: Sin(Declination) array
        :param log_e: Log(Energy/GeV) array
        :param sin_dec_bins: Bins of Sin(Declination) to be used
        :param log_e_bins: bins of log(E/GeV) to be used
        :param weights: Array of weights for events
        :return: Normalised histogram
        """
        hist_2d, _ = np.histogramdd(
            (log_e, sin_dec), bins=(log_e_bins, sin_dec_bins), weights=weights
        )

        return hist_2d

    def create_sig_2d_hist(self, mc, sin_dec_bins, log_e_bins, weight_function):
        """Creates a signal 2D logE/sinDec weighted histogram.

        :param mc: MC Simulations
        :param weight_function: Weight Function
        :return: 2D histogram
        """

        return self.create_2d_hist(
            mc["sinDec"],
            mc["logE"],
            sin_dec_bins,
            log_e_bins,
            weights=weight_function(mc),
        )

    def create_bkg_2d_hist(self, exp, sin_dec_bins, log_e_bins):
        """Creates a background 2D logE/sinDec weighted histogram.
        The weights depend on the chosen background model.

        :param exp: Experimental data (or MC depending on season)
        :return: 2D histogram
        """

        w = self.bkg_weights(exp)
        return self.create_2d_hist(
            exp["sinDec"],
            exp["logE"],
            sin_dec_bins,
            log_e_bins,
            weights=w,
        )

    def create_2d_ratio_hist(
        self,
        exp,
        mc,
        sin_dec_bins,
        log_e_bins,
        weight_f,
    ):
        """Creates a 2D histogram for both data and MC, in which the seasons
        are binned by Sin(Declination) and Log(Energy/GeV). Each histogram is
        normalised in Sin(Declination) bands. Then creates a histogram of the
        ratio of the Signal/Background histograms. In bins where there is
        simulation but no data, a count of 1 is assigned to the background
        histogram.  This is broadly unimportant for unblinding archival searches,
        because there will never be a situation in which a bin without any data
        will be queried. In all other cases, the ratio is set to 1.

        :param exp: Experimental data (or MC) used for bkg hist
        :param mc: MC Simulations used for sig hist
        :param weight_f: Weight Function used for sig hist
        :return: ratio histogram
        """

        bkg_hist = self.create_bkg_2d_hist(exp, sin_dec_bins, log_e_bins)
        sig_hist = self.create_sig_2d_hist(mc, sin_dec_bins, log_e_bins, weight_f)
        n_dimensions = sig_hist.ndim
        norms = np.sum(sig_hist, axis=n_dimensions - 2)
        norms[norms == 0.0] = 1.0
        sig_hist /= norms

        ratio = np.ones_like(bkg_hist, dtype=float)

        for i, bkg_row in enumerate(bkg_hist.T):
            sig_row = sig_hist.T[i]

            fill_mask = (bkg_row == 0.0) & (sig_row > 0.0)
            bkg_row[fill_mask] = 1.0
            # bkg_row /= np.sum(bkg_row)

            mask = (bkg_row > 0.0) & (sig_row > 0.0)
            r = np.ones_like(bkg_row)
            r[mask] = sig_row[mask] / (bkg_row[mask] / np.sum(bkg_row))

            ratio.T[i] = r
        return ratio

    def make_2d_spline_from_hist(
        self, ratio, sin_dec_bins, log_e_bins, smoothing_order: int
    ):

        # Sets bin centers, and order of spline (for x and y)
        sin_bin_center = (sin_dec_bins[:-1] + sin_dec_bins[1:]) / 2.0
        log_e_bin_center = (log_e_bins[:-1] + log_e_bins[1:]) / 2.0

        # Fits a 2D spline function to the log of ratio array
        if smoothing_order == 0:
            # use nearest-neighbor interpolation to ensure that ratio retains the normalization of the underlying histograms
            spline = scipy.interpolate.RegularGridInterpolator(
                (log_e_bin_center, sin_bin_center),
                np.log(ratio),
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )

        # If the splines are of order one the RegularGridInterpolator is used to match the SkyLab behavior
        elif smoothing_order == 1:
            sin_bin_center[0], sin_bin_center[-1] = sin_dec_bins[0], sin_dec_bins[-1]
            # log_e_bins[0], log_e_bins[-1] = log_e_bins[0], log_e_bins[-1]

            binmids = (log_e_bin_center, sin_bin_center)

            spline = scipy.interpolate.RegularGridInterpolator(
                binmids,
                np.log(ratio),
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

        # If the interpolating splines are of order greater than 1, use RectBivariateSpline
        else:
            # This is order-th order in both dimensions
            spline = scipy.interpolate.RectBivariateSpline(
                log_e_bin_center,
                sin_bin_center,
                np.log(ratio),
                kx=smoothing_order,
                ky=smoothing_order,
                s=0,
            )

        return spline

    def create_2d_ratio_spline(
        self, exp, mc, sin_dec_bins, log_e_bins, weight_f, smoothing_order: int
    ):
        """Creates 2D histograms for both data and MC, in which the seasons
        are binned by Sin(Declination) and Log(Energy/GeV). Each histogram is
        normalised in Sin(Declination) bands. Then creates a histogram of the
        ratio of the Signal/Background histograms. In bins where there is
        simulation but no data, the ratio is set to the highest ratio
        value found for cases with both data and MC.  This is broadly
        unimportant for unblinded archival searches, because there will never
        be a situation in which a bin without any data will be queried. In all
        other cases, the ratio is set to 1.

        A 2D spline, of chosen order in x and y, is then fit to the Log(Ratio),
        and returned.

        :param exp: Experimental data (or MC) used for bkg hist
        :param mc: MC Simulations used for sig hist
        :param weight_f: weight function used for sig hist
        :param bkg_weights: array of weights for bkg events
        :param smoothing_order: order of the spline
        :return: 2D spline function
        """

        ratio = self.create_2d_ratio_hist(
            exp,
            mc,
            sin_dec_bins,
            log_e_bins,
            weight_f,
        )

        spline = self.make_2d_spline_from_hist(
            ratio, sin_dec_bins, log_e_bins, smoothing_order
        )

        return spline

    def create_gamma_2d_ratio_spline(
        self, exp, mc, sin_dec_bins, log_e_bins, gamma, smoothing_order: int
    ):
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

        return self.create_2d_ratio_spline(
            exp, mc, sin_dec_bins, log_e_bins, weight_function, smoothing_order
        )

    def create_2d_splines(self, exp, mc, sin_dec_bins, log_e_bins):
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

        gamma_support_points = get_gamma_support_points(self.gamma_precision)
        for gamma in gamma_support_points:
            splines[gamma] = self.create_gamma_2d_ratio_spline(
                exp, mc, sin_dec_bins, log_e_bins, gamma, self.smoothing_order
            )

        if not np.any(list(splines.values())):
            logger.warning("No splines!")
            return

        return splines

    @staticmethod
    def make_plot(
        hist,
        savepath,
        x_bins,
        y_bins,
        normed=True,
        log_min=5,
        label_x=r"$\sin(\delta)$",
        label_y="log(Energy)",
    ):
        """Plot a 2D histogram"""
        if normed:
            norms = np.sum(hist, axis=hist.ndim - 2)
            norms[norms == 0.0] = 1.0
            hist /= norms
        else:
            hist = np.log(np.array(hist))
        plt.figure()
        ax = plt.subplot(111)
        X, Y = np.meshgrid(x_bins, y_bins)
        if not normed:
            max_col = min(
                abs(min([min(row) for row in hist.T])),
                max([max(row) for row in hist.T]),
            )
            cbar = ax.pcolormesh(X, Y, hist, cmap="seismic", vmin=-5, vmax=5)
            plt.colorbar(cbar, label="Log(Signal/Background)")
        else:
            hist[hist == 0.0] = np.nan
            cbar = ax.pcolormesh(X, Y, hist)
            plt.colorbar(cbar, label="Column-normalised density")
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.savefig(savepath)
        plt.close()

    def make_individual_spline_set(self, season, SoB_path):
        """Create the SoB splines dictionary for given season,
        and plot the normalized 2D histograms for background
        and signal, as well as the log(S/B) 2D hist.
        The latter two are plotted for gamma values in
        np.linspace(1.0, 4.0, 7).
        """
        try:
            logger.info(f"Making SoB splines for {season.season_name}")

            exp = season.get_background_model()
            mc = season.get_pseudo_mc()

            sin_dec_bins = season.sin_dec_bins
            log_e_bins = season.log_e_bins

            ##### MAKE SPLINES #####
            splines = self.create_2d_splines(exp, mc, sin_dec_bins, log_e_bins)

            logger.info(f"Saving SoB splines to {SoB_path}.")

            try:
                os.makedirs(os.path.dirname(SoB_path))
            except OSError:
                pass

            with open(SoB_path, "wb") as f:
                Pickle.dump(splines, f)

            if isinstance(splines, type(None)):
                return

            ##### GENERATE PLOTS #####
            base_plot_path = get_base_sob_plot_dir(season)
            sfx = f"{self.spline_name}_" if self.spline_name != "no_difffuse" else ""

            # plot 2D background hist
            bkg_hist = self.create_bkg_2d_hist(exp, sin_dec_bins, log_e_bins)
            self.make_plot(
                bkg_hist,
                savepath=base_plot_path + sfx + "bkg.pdf",
                x_bins=sin_dec_bins,
                y_bins=log_e_bins,
            )

            # plot 2D signal & S/B hist, and spline
            for gamma in np.linspace(1.0, 4.0, 7):
                plot_path = (
                    base_plot_path
                    + "gamma="
                    + str(gamma)
                    + "/"
                    + f"precision{get_gamma_precision(self.gamma_precision)}_"
                    f"smoothing{self.smoothing_order}" + "/"
                )

                try:
                    os.makedirs(plot_path)
                except OSError:
                    pass

                def weight_function(sig_mc):
                    return energy_pdf.weight_mc(sig_mc, gamma)

                mc_hist = self.create_sig_2d_hist(
                    mc, sin_dec_bins, log_e_bins, weight_function
                )
                self.make_plot(
                    mc_hist,
                    plot_path + sfx + "sig.pdf",
                    sin_dec_bins,
                    log_e_bins,
                )

                self.make_plot(
                    self.create_2d_ratio_hist(
                        exp, mc, sin_dec_bins, log_e_bins, weight_function
                    ),
                    plot_path + sfx + "SoB.pdf",
                    sin_dec_bins,
                    log_e_bins,
                    normed=False,
                )

                Z = []
                for s in sin_dec_bins:
                    z_line = []
                    for e in log_e_bins:
                        # logging.debug(f'{e}, {s}')
                        try:
                            z_line.append(splines[gamma](e, s)[0][0])
                        except:
                            z_line.append(splines[gamma]((e, s)))
                    Z.append(z_line)

                Z = np.array(Z).T

                max_col = min(
                    abs(min([min(row) for row in Z])), max([max(row) for row in Z])
                )

                plt.figure()
                ax = plt.subplot(111)
                X, Y = np.meshgrid(sin_dec_bins, log_e_bins)
                cbar = ax.pcolormesh(
                    X, Y, Z, cmap="seismic", vmin=-max_col, vmax=max_col, shading="auto"
                )
                plt.colorbar(cbar, label="Log(Signal/Background)")
                plt.xlabel(r"$\sin(\delta)$")
                plt.ylabel("log(Energy)")
                plt.savefig(plot_path + sfx + "spline.pdf")
                plt.close()

            del mc

        except IOError:
            pass

    def make_splines(self, seasons):
        """Make the S/B splines for all dataset seasons,
        as well as the background spatial spline.
        """

        logger.info(
            "Splines will be made to calculate the Signal/Background ratio of "
            + "the MC to data. The MC will be weighted with a power law for the signal, "
            + f"for each gamma in: {list(get_gamma_support_points(self.gamma_precision))}. "
            + "For the background, the MC is weighted according to the model assumption"
        )

        for season in seasons.values():
            SoB_path = SoB_spline_path(season, **self.sob_dict)
            self.make_individual_spline_set(season, SoB_path)
            self.make_background_spline(season)

    def load_spline(self):
        path = SoB_spline_path(self.season, **self.sob_dict)

        logger.debug(f"Loading SoB spline from {path}")

        try:
            with open(path, "rb") as f:
                res = Pickle.load(f)
        except FileNotFoundError as err:
            logger.info(
                f"No cached spline found at {path}. Creating this file instead."
            )
            logger.info(f"Cause: {err}")
            self.make_individual_spline_set(self.season, path)
            with open(path, "rb") as f:
                res = Pickle.load(f)

        except ModuleNotFoundError as err:
            logger.error(
                "A spline was found but it seems incompatible with this installation."
            )
            logger.error(f"Cause: {err}")
            raise

        return res


@SoB_splines.register_subclass("no_difffuse")
class NoDiffuseSpline(SoB_splines):
    """Atmospheric-only background model.
    The weights for creating the background
    2D histogram that goes into creating
    the SoB splines are simply the
    conventional atmospheric weights in the MC
    """

    def bkg_weights(self, exp: Table) -> np.ndarray:
        logger.debug("Atmospheric-only background weights")
        return np.asarray(exp["weight"])


@SoB_splines.register_subclass("spl_difffuse")
class SPLDiffuseSpline(SoB_splines):
    """Single powerlaw diffuse + atmospheric background.
    Provided the (per-flavour) best-fit spectral parameters
    phi & gamma, the per-flavour diffuse flux following SPL is
    Phi_nu_antinu = phi * (E/100 TeV)**-gamma * 1e-18 /GeV cm2 s sr
    The flux is multiplied with the livetime
    to get the fluence and then with "ow"
    in order to get the weights that go into
    the background histogram.
    The histogram is weighted w/ atmospheric + diffuse weights,
    where atmospheric weights are the
    conventional atmospheric weights in the MC.
    If not provided, the default SPL parameters
    are taken from the 10y ESTES SPL
    (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.022001)
    w/ best-fit phi_0 = 1.68 & gamma = 2.58
    """

    def __init__(self, season, SoB_dict) -> None:
        super().__init__(season, SoB_dict)

        time_pdf = self.season.get_time_pdf()
        self.livetime = time_pdf.get_livetime() * 3600 * 24

    def get_diffuse_flux(self, exp: Table) -> np.ndarray:
        phi = self.sob_dict.get("phi0", 1.68)
        gamma = self.sob_dict.get("spl_gamma", 2.58)
        logger.info(
            f"SPL diffuse flux used with best-fit phi = {phi} & gamma = {gamma}"
        )
        return 1e-18 * phi * (np.asarray(exp["trueE"]) / 1e5) ** -gamma  # /GeV cm2 s sr

    def bkg_weights(self, exp: Table) -> np.ndarray:
        logger.debug("Background weights w/ SPL diffuse flux")
        flux = self.get_diffuse_flux(exp)
        fluence = flux * self.livetime
        diff_weights = fluence * np.asarray(exp["ow"])
        return np.array(diff_weights + exp["weight"])


@SoB_splines.register_subclass("bpl_difffuse")
class BPLDiffuseSpline(SPLDiffuseSpline):
    """Broken powerlaw diffuse + atmospheric background.
    Provided the (per-flavour) best-fit spectral parameters
    phi, gamma1, gamma2, and E_break the
    per-flavour diffuse flux following a BPL is
    Phi_nu_antinu = C * phi * (E_break/100 TeV)**-gamma1 * (E/E_break)**-gamma1
    for E < E_break & E_break > 100 TeV, and
    Phi_nu_antinu = C * phi * (E_break/100 TeV)**-gamma2 * (E/E_break)**-gamma2
    for E >= E_break & E_break <= 100 TeV,
    where C = 10**-18 /GeV cm2 s sr.
    The flux is multiplied with the livetime
    to get the fluence and then with "ow"
    in order to get the weights that go into
    the background histogram.
    The histogram is weighted w/ atmospheric + diffuse weights,
    where atmospheric weights are the
    conventional atmospheric weights in the MC.
    If not provided, the default BPL parameters
    are taken from the 11y MESE BPL
    (https://arxiv.org/abs/2507.22233), chosen against
    the Combined Fit (CF) since it has a higher TS (see table 3),
    w/ best-fit phi_0 = 2.28, gamma1 = 1.72, gamma2 = 2.84, E_break = 33.11 TeV
    """

    def __init__(self, season, SoB_dict) -> None:
        super().__init__(season, SoB_dict)

    def get_diffuse_flux(
        self, exp: Table
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi0 = self.sob_dict.get("phi0", 2.28)
        gamma1 = self.sob_dict.get("gamma1", 1.72)
        gamma2 = self.sob_dict.get("gamma2", 2.84)
        log_Ebreak = self.sob_dict.get("logEbreak", 4.52)  # log(E_break/GeV)
        logger.info(
            f"BPL diffuse flux used with best-fit phi = {phi0}, gamma1 = {gamma1}, "
            + f"gamma2 = {gamma2}, and E_break = {10**log_Ebreak/1e3} TeV"
        )
        E_break = 10**log_Ebreak  # in GeV
        if E_break <= 1e5:
            phi = 1e-18 * phi0 * (E_break / 1e5) ** -gamma2
        else:
            phi = 1e-18 * phi0 * (E_break / 1e5) ** -gamma1
        mask1 = np.nonzero(exp["trueE"] < E_break)
        mask2 = np.nonzero(exp["trueE"] >= E_break)
        phi1 = (
            phi * (np.asarray(exp[mask1]["trueE"]) / E_break) ** -gamma1
        )  # /GeV cm2 s sr
        phi2 = (
            phi * (np.asarray(exp[mask2]["trueE"]) / E_break) ** -gamma2
        )  # /GeV cm2 s sr
        return phi1, mask1, phi2, mask2

    def bkg_weights(self, exp: Table) -> np.ndarray:
        logger.debug("Background weights w/ BPL diffuse flux")
        w = np.empty(len(exp), dtype=float)
        flux1, mask1, flux2, mask2 = self.get_diffuse_flux(exp)

        # get diffuse + atm weights when E < E_break
        fluence1 = flux1 * self.livetime
        bpl1_w = fluence1 * np.asarray(exp[mask1]["ow"])
        w1 = np.array(bpl1_w + exp[mask1]["weight"])

        # get diffuse + atm weights when E >= E_break
        fluence2 = flux2 * self.livetime
        bpl2_w = fluence2 * np.asarray(exp[mask2]["ow"])
        w2 = np.array(bpl2_w + exp[mask2]["weight"])

        w[mask1] = w1
        w[mask2] = w2
        return w


# def delete_old_splines():
#     """Deletes previously produced splines of the SoB energy PDF histogram, the spatial background PDF and the
#     acceptance function"""
#     logging.info('Deleting old splines!')
#     directories_to_clear = [SoB_spline_dir, bkg_spline_dir, acc_f_dir]
#     for d in directories_to_clear:
#         logging.debug(f'clearing {d}')
#         shutil.rmtree(d)
#         os.mkdir(d)


# def use_precision(mode='flarestack'):
#     """
#     Configures the current environment to use the desired precision in gamma.
#     Deletes previously produced splines if precision changes.
#     :param mode: float or 'flarestack' or 'skyLab', default:'flarestack'
#     """
#
#     old_precision = os.environ.get(environment_precision_key, None)
#     new_precision = '0.025' if mode in ['flaresatck', 'Flarestack', 'default'] else \
#         '0.1' if (mode in ['SkyLab', 'skylab', 'skylab_splines']) or ('skylab_splines' in mode) else \
#         mode if isinstance(mode, float) else \
#         None
#
#     if not new_precision:
#         logger.warning(f'Mode {mode} not known! Use "Flarestack", "SkyLab" or a float '
#                        f'to specify the order of the interpolating spline.')
#
#     if not new_precision or (new_precision != old_precision):
#         logger.info(f'Gamma precision has changed from {old_precision} to {new_precision}')
#         delete_old_splines()
#         os.environ[environment_precision_key] = str(new_precision)
#     else:
#         logging.info(f'New precision {new_precision} same as old precision {old_precision}.')
#
#     logging.info(f'Gamma precision is now {os.environ[environment_precision_key]}')


# def use_smoothing(mode='flarestack'):
#     """
#     Configures the current environment to use the desired smoothing order when building energy PDFs
#     Deletes previously produced splines if precision changes.
#     :param mode: int or 'flarestack' or 'skyLab', default:'flarestack'
#     """
#
#     old_smoothing_order = os.environ.get(environment_smoothing_key, None)
#     new_smoothing_order = '2' if mode in ['default', 'flarestack', 'Flarestack'] else \
#         '1' if mode in ['SkyLab', 'skylab', 'skylab_splines'] else \
#         str(mode) if isinstance(mode, int) else \
#         None
#
#     if not new_smoothing_order:
#         logger.warning(f'Mode {mode} not known! Use "Flarestack", "SkyLab" or an integer '
#                        f'to specify the order of the interpolating spline.')
#
#     if not old_smoothing_order or (old_smoothing_order != new_smoothing_order):
#         logger.info(f'Smoothing order changed from {old_smoothing_order} to {new_smoothing_order}')
#         delete_old_splines()
#         os.environ[environment_smoothing_key] = str(new_smoothing_order)
#     else:
#         logging.info(f'New PDF smoothing order {new_smoothing_order} is the same as old one {old_smoothing_order}')
#
#     logging.info(f'Smoothing order is now {os.environ[environment_smoothing_key]}')

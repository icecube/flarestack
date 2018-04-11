import numpy as np
import scipy.interpolate
from energy_PDFs import EnergyPDF


class SoB:

    def __init__(self, season, splines=None, **kwargs):
        self.season = season

        # Bins for sin declination (not evenly spaced)
        self.sin_dec_bins = np.unique(np.concatenate([
            np.linspace(-1., -0.9, 2 + 1),
            np.linspace(-0.9, -0.2, 8 + 1),
            np.linspace(-0.2, 0.2, 15 + 1),
            np.linspace(0.2, 0.9, 12 + 1),
            np.linspace(0.9, 1., 2 + 1),
        ]))

        # If provided in kwargs, sets whether the spectral index (gamma)
        # should be included as a fit prameter. If this is not specified,
        # the default is to not fit gamma.
        try:
            self.fit_gamma = kwargs["Fit Gamma?"]
        except KeyError:
            self.fit_gamma = False

        e_pdf_dict = kwargs["LLH Energy PDF"]

        if e_pdf_dict is not None:
            self.energy_pdf = EnergyPDF.create(e_pdf_dict)
            # Bins for energy Log(E/GeV)
            self.energy_bins = np.linspace(1., 10., 40 + 1)

            # Sets precision for energy SoB
            self.precision = .1

            # If there is an LLH energy pdf specified, uses that gamma as the
            # default for weighting the detector acceptance.
            self.default_gamma = self.energy_pdf.gamma

            # If gamma is to be fit, the SoB energy histograms must be
            # evaluated for a range of values between the bounds of 1 and 4
            if self.fit_gamma:
                # Produces a set (i.e no duplicates) of datapoints for gamma
                # This is best on 33 points between 0.9 and 4.1
                # Each point is modified by _around(i)
                # Useful for different precisions, where rounding errors might
                # otherwise lead to duplicates in set
                self.gamma_support_points = set(
                    [self._around(i) for i in np.linspace(0.9, 4.1, 30 + 3)])

        # Checks gamma is not being fit without an energy PDF provided
        elif self.fit_gamma:
            raise Exception("LLH has been set to fit gamma, "
                            "but no Energy PDF has been provided")

        # If gamma is not a fit parameter, and no energy PDF has been
        # provided, sets a default value of gamma = 2.
        else:
            self.default_gamma = 2.

        if splines is None:

            self._mc = np.load(season["mc_path"])
            self._exp = np.load(season["exp_path"])

            self.bkg_spatial = self.create_bkg_spatial_spline(self._exp)

            if e_pdf_dict is not None:
                print "Making Log(Signal/Background) Splines."
                self.SoB_spline_2Ds = self.create_2d_splines()
                print "Made", len(self.SoB_spline_2Ds), "Splines."

        else:
            self.SoB_spline_2Ds = splines["SoB_spline_2D"]
            self.bkg_spatial = splines["Background spatial"]

    def _around(self, value):
        """Produces an array in which the precision of the value
        is rounded to the nearest integer. This is then multiplied
        by the precision, and the new value is returned.

        :param value: value to be processed
        :return: value after processed
        """
        return np.around(float(value) / self.precision) * self.precision

    def create_bkg_spatial_spline(self, exp):
        """Creates the spatial PDF for background.
        Generates a histogram for the exp. distribution in sin declination.
        Fits a spline function to the distribution, giving a spatial PDF.
        Returns this spatial PDF.

        :param exp: Experimental data (background)
        :return: Background spline function
        """
        sin_dec_bins = self.sin_dec_bins
        sin_dec_range = (np.min(sin_dec_bins), np.max(sin_dec_bins))
        hist, bins = np.histogram(
            exp['sinDec'], density=True, bins=sin_dec_bins, range=sin_dec_range)

        bins = np.concatenate([bins[:1], bins, bins[-1:]])
        hist = np.concatenate([hist[:1], hist, hist[-1:]])

        bkg_spline = scipy.interpolate.InterpolatedUnivariateSpline(
                                (bins[1:] + bins[:-1]) / 2.,
                                np.log(hist), k=2)
        return bkg_spline

    def create_2d_hist(self, sin_dec, log_e, weights):
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
        energy_bins = self.energy_bins
        sin_dec_bins = self.sin_dec_bins
        # Produces the histogram
        hist_2d, binedges = np.histogramdd(
            (log_e, sin_dec), bins=(energy_bins, sin_dec_bins), weights=weights)
        n_dimensions = hist_2d.ndim

        # Normalises histogram
        norms = np.sum(hist_2d, axis=n_dimensions - 2)
        norms[norms == 0.] = 1.
        hist_2d /= norms

        return hist_2d

    def create_2d_ratio_spline(self, gamma):
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

        bkg_hist = self.create_2d_hist(self._exp["sinDec"], self._exp["logE"],
                                       weights=np.ones_like(self._exp["logE"]))

        sig_hist = self.create_2d_hist(np.sin(self._mc["trueDec"]),
                                       self._mc["logE"],
                                       weights=self.energy_pdf.weight_mc(
                                           self._mc, gamma))

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
        sin_bin_center = (self.sin_dec_bins[:-1] + self.sin_dec_bins[1:]) / 2.
        log_e_bin_center = (self.energy_bins[:-1] + self.energy_bins[1:]) / 2.
        order = 2

        # Fits a 2D spline function to the log of ratio array
        # This is 2nd order in both dimensions
        spline = scipy.interpolate.RectBivariateSpline(
            log_e_bin_center, sin_bin_center, np.log(ratio),
            kx=order, ky=order, s=0)

        return spline

    def create_2d_splines(self):
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

        if self.fit_gamma:
            for gamma in self.gamma_support_points:
                splines[gamma] = self.create_2d_ratio_spline(gamma)

        else:
            gamma = self.default_gamma
            splines[gamma] = self.create_2d_ratio_spline(gamma)

        return splines

# ==============================================================================
# Optional Functions for Plotting
# ==============================================================================

    def plot_2d_ratio_spline(self):
        """Creates a plot of the 2D spline function over the Log(Energy) &
        Sin(Declination) range, and saves this as a PDF.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        n_step = 100
        e_range = np.linspace(self.energy_bins[0], self.energy_bins[-1], n_step)
        sin_dec_range = np.linspace(-1, 1., n_step)

        z = []

        for sin_dec in sin_dec_range:
            z.append(list(self.SoB_spline_2D(e_range, sin_dec).T.tolist()[0]))

        z = np.array(z).T

        plt.figure()
        ax = plt.subplot(111)
        cbar = plt.pcolormesh(sin_dec_range, e_range, z,
                              cmap=cm.get_cmap('seismic'),
                              label='Log(Ratio)')
        plt.axis([sin_dec_range[0], sin_dec_range[-1], e_range[0], e_range[-1]])
        ax.set_ylabel("Log(Energy/GeV)")
        ax.set_xlabel("Sin(Declination)")
        plt.colorbar(cbar)
        plt.title("Log(Ratio) of Signal/Background for " + self.season["Name"])
        plt.savefig("energy_vs_sin_dec" + self.season["Name"] + ".pdf")
        plt.close()

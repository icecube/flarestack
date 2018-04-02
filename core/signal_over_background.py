import numpy as np
import scipy
from energy_PDFs import EnergyPDF


class SoB:

    def __init__(self, season, **kwargs):
        self.season = season
        self.energy_pdf = EnergyPDF.create(kwargs["LLH Energy PDF"])
        self._mc = np.load(season["mc_path"])
        self.mc_weights = self.energy_pdf.weight_mc(self._mc)
        self._exp = np.load(season["exp_path"])

        # Bins for energy Log(E/GeV)
        self.energy_bins = np.linspace(1., 10., 40 + 1)
        # Bins for sin declination (not evenly spaced)
        self.sinDec_bins = np.unique(np.concatenate([
                                np.linspace(-1., -0.9, 2 + 1),
                                np.linspace(-0.9, -0.2, 8 + 1),
                                np.linspace(-0.2, 0.2, 15 + 1),
                                np.linspace(0.2, 0.9, 12 + 1),
                                np.linspace(0.9, 1., 2 + 1),
                                ]))

        self.spline_2D = self.create_2d_ratio_spline()

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
        sin_dec_bins = self.sinDec_bins
        # Produces the histogram
        hist_2d, binedges = np.histogramdd(
            (log_e, sin_dec), bins=(energy_bins, sin_dec_bins), weights=weights)
        n_dimensions = hist_2d.ndim

        # Normalises histogram
        norms = np.sum(hist_2d, axis=n_dimensions - 2)
        norms[norms == 0.] = 1.
        hist_2d /= norms

        return hist_2d

    def create_2d_ratio_spline(self):

        bkg_hist = self.create_2d_hist(self._exp["sinDec"], self._exp["logE"],
                                       weights=np.ones_like(self._exp["logE"]))

        sig_hist = self.create_2d_hist(np.sin(self._mc["trueDec"]),
                                       self._mc["logE"],
                                       weights=self.mc_weights)

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

        # Finds the minimum ratio
        min_ratio = np.amax(ratio)
        # Where true in sig and false in bkg, sets ratio to minimum ratio
        np.copyto(ratio, min_ratio, where=domain_sig & ~domain_bkg)

        # Sets Bin centers, and order of spline (for x and y)
        sin_dec_bin_center = (self.sinDec_bins[:-1] + self.sinDec_bins[1:]) / 2.
        log_e_bin_center = (self.energy_bins[:-1] + self.energy_bins[1:]) / 2.
        log_e_order = 2

        # Fits a 2D spline function to the log of ratio array
        # This is 2nd order in both dimensions
        spline = scipy.interpolate.RectBivariateSpline(
            log_e_bin_center, sin_dec_bin_center, np.log(ratio),
            kx=log_e_order, ky=log_e_order, s=0)

        return spline

    def plot_2d_ratio_spline(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        n_step = 100
        e_range = np.linspace(self.energy_bins[0], self.energy_bins[-1], n_step)
        sinDec_range = np.linspace(-1, 1., n_step)

        z = []

        for sinDec in sinDec_range:
            z.append(list(self.spline_2D(e_range, sinDec).T.tolist()[0]))
            # print z
            # raw_input("prompt")

        z = np.array(z).T

        plt.figure()
        ax = plt.subplot(111)
        cbar = plt.pcolormesh(sinDec_range, e_range, z,
                              cmap=cm.get_cmap('seismic'),
                              label='Log(Ratio)')
        plt.axis([sinDec_range[0], sinDec_range[-1], e_range[0], e_range[-1]])
        ax.set_ylabel("Log(Energy/GeV)")
        ax.set_xlabel("Sin(Declination)")
        plt.colorbar(cbar)
        plt.title("Log(Ratio) of Signal/Background for " + self.season["Name"])
        plt.savefig("energy_vs_cos_zen_" + self.season["Name"] + ".pdf")
        plt.close()


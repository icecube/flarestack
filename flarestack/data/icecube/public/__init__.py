import os
import numpy as np
import pickle
import scipy.interpolate
from flarestack.data import SeasonWithoutMC
from flarestack.icecube_utils.dataset_loader import data_loader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from flarestack.shared import eff_a_plot_dir, energy_proxy_path, \
    med_ang_res_path


class PublicICSeason(SeasonWithoutMC):

    def __init__(self, season_name, sample_name, exp_path, pseudo_mc_path,
                 sin_dec_bins, log_e_bins, **kwargs):
        SeasonWithoutMC.__init__(self, season_name, sample_name, exp_path,
                                 pseudo_mc_path, **kwargs)
        self.sin_dec_bins = sin_dec_bins
        self.log_e_bins = log_e_bins

    def load_data(self, path, **kwargs):
        return data_loader(path, **kwargs)

    def load_angular_resolution(self):
        path = med_ang_res_path(self)

        with open(path, "rb") as f:
            [x, y] = pickle.load(f)

        map_f = scipy.interpolate.interp1d(x, y)

        return lambda log_e: map_f(log_e)

    def load_effective_area(self):
        pseudo_mc = self.get_pseudo_mc(cut_fields=False)

        log_e_bin_center = sorted(set(pseudo_mc["logE"]))
        sin_bin_center = sorted(set(pseudo_mc["sinDec"]))

        eff_a = pseudo_mc
        eff_a = np.reshape(eff_a, (len(log_e_bin_center), len(sin_bin_center),))

        order = 1

        effective_area_spline = scipy.interpolate.RectBivariateSpline(
            log_e_bin_center, sin_bin_center, np.log(eff_a["a_eff"] + 1e-9),
            kx=order, ky=order, s=0)

        return lambda x, y: np.exp(effective_area_spline.ev(x, y))

    def load_energy_proxy_mapping(self):
        path = energy_proxy_path(self)

        with open(path, "rb") as f:
            [x, y] = pickle.load(f)

        map_f = scipy.interpolate.interp1d(x, y)

        return lambda e: np.exp(map_f(np.log10(e)))

    def plot_effective_area(self):

        savepath = eff_a_plot_dir + self.sample_name + "/" + self.season_name \
                   + ".pdf"

        try:
            os.makedirs(os.path.dirname(savepath))
        except OSError:
            pass

        plt.figure()
        ax = plt.subplot(111)
        X, Y = np.meshgrid(self.log_e_bins, self.sin_dec_bins,)

        eff_a_f = self.load_effective_area()

        vals = eff_a_f(X, Y)

        cbar = ax.pcolormesh(X, Y, vals, norm=LogNorm())
        cb = plt.colorbar(cbar, label="Effective Area [m]", ax=ax)
        plt.ylabel(r"$\sin(\delta)$")

        locs, labels = plt.xticks()
        labels = [10**float(item) for item in locs]
        plt.xticks(locs, labels)

        f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % 10**x))
        fmt = ticker.FuncFormatter(g)
        ax.xaxis.set_major_formatter(fmt)
        plt.xlabel(r"$E_{\nu}$")
        print("Saving to", savepath)
        plt.savefig(savepath)
        plt.close()

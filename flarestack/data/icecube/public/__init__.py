import os
import numpy as np
import pickle
import csv
import scipy.interpolate
from flarestack.data import SeasonWithoutMC
from flarestack.icecube_utils.dataset_loader import data_loader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from flarestack.shared import eff_a_plot_dir, energy_proxy_path, \
    med_ang_res_path, energy_proxy_plot_path


class PublicICSeason(SeasonWithoutMC):

    def __init__(self, season_name, sample_name, exp_path, pseudo_mc_path,
                 sin_dec_bins, log_e_bins, a_eff_path, **kwargs):
        SeasonWithoutMC.__init__(self, season_name, sample_name, exp_path,
                                 pseudo_mc_path, **kwargs)
        self.sin_dec_bins = sin_dec_bins
        self.log_e_bins = log_e_bins
        self.a_eff_path = a_eff_path

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

    def plot_effective_area(self, show=False):

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
        if show:
            plt.show()
        else:
            plt.close()

    def get_raw_pseudo_mc(self):

        data_dtype = np.dtype([
            ('logE', np.float),
            ('trueE', np.float),
            ('sinDec', np.float),
            ('trueDec', np.float),
            ('ow', np.float),
            ('a_eff', np.float),
            ("sigma", np.float)
        ])

        pseudo_mc = []

        with open(self.a_eff_path, "r") as f:

            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):

                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    true_e = 0.5*(row[0] + row[1])
                    log_e = np.log10(true_e)
                    sin_dec = -0.5*(row[2] + row[3])
                    true_dec = np.arcsin(sin_dec)
                    a_eff = row[4]

                    entry = tuple([
                        log_e, true_e, sin_dec, true_dec,
                        a_eff, a_eff, np.nan
                    ])

                    pseudo_mc.append(entry)

        pseudo_mc = np.array(pseudo_mc, dtype=data_dtype)

        return pseudo_mc

    def map_energy_proxy(self, show=False):

        exp = self.get_background_model()

        pseudo_mc = self.get_raw_pseudo_mc()

        # Select only upgoing muons. For these events, the dominant
        # background is atmospheric neutrinos with a known spectrum of E^-3.7.
        # Downgoing events, on the other hand, are contaminated by sneaking
        # muon bundles which are harder to model.

        exp = exp[exp["sinDec"] > 0.]

        plt.figure()
        ax1 = plt.subplot(311)
        res = ax1.hist(exp["logE"], density=True)
        ax1.set_title("Energy Proxy (Data)")
        ax1.set_xlabel(r"$\log_{10}(E_{proxy})$")

        exp_vals = res[0]
        exp_bins = res[1]
        ax1.set_yscale("log")
        ax2 = plt.subplot(312, sharex=ax1)
        res = ax2.hist(
            pseudo_mc["logE"],
            weights=pseudo_mc["ow"] * pseudo_mc["trueE"] ** -3.7,
            density=True, bins=exp_bins)
        mc_vals = res[0]

        ax2.set_yscale("log")
        ax2.set_title(r"Expected True Energy ($E^{-3.7}$)")
        ax2.set_xlabel(r"$\log_{10}(E_{true})$")

        # Maps ratio of expected neutrino energies to energy proxy values
        # This can tell us about how true energy maps to energy proxy

        centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])

        # Fill in empty bins

        mc_vals = np.array(mc_vals)

        mc_vals += min(pseudo_mc["ow"][pseudo_mc["ow"] > 0.]) * centers ** -3.7

        x = [-5.0] + list(centers) + [15.0]
        y = exp_vals / mc_vals
        y = [y[0]] + list(y) + [y[-1]]

        log_e_weighting = scipy.interpolate.interp1d(x, np.log(y))

        ax3 = plt.subplot(313)
        plt.plot(centers, exp_vals / mc_vals)
        plt.plot(centers, np.exp(log_e_weighting(centers)),
                 linestyle=":")
        ax3.set_yscale("log")
        ax3.set_title("Ratio")
        ax3.set_xlabel(r"$\log_{10}(E)$")

        plt.tight_layout()

        save_path = energy_proxy_plot_path(self)

        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError:
            pass

        print("Saving to", save_path)

        plt.savefig(save_path)

        pseudo_mc["ow"] *= np.exp(log_e_weighting(pseudo_mc["logE"]))

        mc_path = self.pseudo_mc_path

        np.save(mc_path, pseudo_mc)

        ep_path = energy_proxy_path(self)

        try:
            os.makedirs(os.path.dirname(ep_path))
        except OSError:
            pass

        with open(ep_path, "wb") as f:
            print("Saving converted numpy array to", ep_path)
            pickle.dump([x, np.log(y)], f)

        if show:
            plt.show()
        else:
            plt.close()
import os
import numpy as np
import pickle
import csv
import scipy.interpolate
import logging
from astropy.table import Table
from flarestack.data import SeasonWithoutMC, Season
from flarestack.icecube_utils.dataset_loader import data_loader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from flarestack.shared import (
    eff_a_plot_dir,
    energy_proxy_path,
    med_ang_res_path,
    energy_proxy_plot_path,
)

logger = logging.getLogger(__name__)


class PublicICSeason(SeasonWithoutMC):
    def __init__(
        self,
        season_name,
        sample_name,
        exp_path,
        pseudo_mc_path,
        sin_dec_bins,
        log_e_bins,
        a_eff_path,
        **kwargs,
    ):
        SeasonWithoutMC.__init__(
            self, season_name, sample_name, exp_path, pseudo_mc_path, **kwargs
        )
        self.sin_dec_bins = sin_dec_bins
        self.log_e_bins = log_e_bins
        self.a_eff_path = a_eff_path

    def load_data(self, path, **kwargs) -> Table:
        return data_loader(path, **kwargs)

    def load_angular_resolution(self):
        path = med_ang_res_path(self)

        with open(path, "rb") as f:
            [x, y] = pickle.load(f)

        map_f = scipy.interpolate.interp1d(x, y)

        return lambda log_e: map_f(log_e)

    def load_effective_area(self):
        pseudo_mc = self.get_pseudo_mc(cut_fields=False)

        entry_0 = pseudo_mc[0]

        log_e_bin_center = list(
            pseudo_mc[pseudo_mc["sinDec"] == entry_0["sinDec"]]["logE"]
        )

        # Make sure values are strictly increasing

        if log_e_bin_center != list(sorted(set(pseudo_mc["logE"]))):
            x_sign = -1.0
            log_e_bin_center = list(sorted(set(pseudo_mc["logE"])))
        else:
            x_sign = 1.0

        sin_bin_center = list(pseudo_mc[pseudo_mc["logE"] == entry_0["logE"]]["sinDec"])

        if sin_bin_center != list(sorted(set(pseudo_mc["sinDec"]))):
            y_sign = -1.0
            sin_bin_center = list(sorted(set(pseudo_mc["sinDec"])))
        else:
            y_sign = 1.0

        eff_a = pseudo_mc

        eff_a = np.reshape(
            eff_a,
            (
                len(log_e_bin_center),
                len(sin_bin_center),
            ),
        )

        order = 1

        effective_area_spline = scipy.interpolate.RectBivariateSpline(
            log_e_bin_center,
            sin_bin_center,
            np.log(eff_a["a_eff"] + 1e-9),
            kx=order,
            ky=order,
            s=0,
        )

        return lambda x, y: np.exp(effective_area_spline.ev(x * x_sign, y * y_sign))

    def load_energy_proxy_mapping(self):
        path = energy_proxy_path(self)

        with open(path, "rb") as f:
            [x, y] = pickle.load(f)

        map_f = scipy.interpolate.interp1d(x, y)

        return lambda e: np.exp(map_f(np.log10(e)))

    def plot_effective_area(self, show=False):
        savepath = eff_a_plot_dir + self.sample_name + "/" + self.season_name + ".pdf"

        try:
            os.makedirs(os.path.dirname(savepath))
        except OSError:
            pass

        plt.figure()
        ax = plt.subplot(111)
        X, Y = np.meshgrid(
            self.log_e_bins,
            self.sin_dec_bins,
        )

        eff_a_f = self.load_effective_area()

        vals = eff_a_f(X, Y)

        cbar = ax.pcolormesh(X, Y, vals, norm=LogNorm(), shading="auto")
        cb = plt.colorbar(cbar, label="Effective Area [m]", ax=ax)
        plt.ylabel(r"$\sin(\delta)$")

        locs, labels = plt.xticks()
        labels = [10 ** float(item) for item in locs]
        plt.xticks(locs, labels)
        plt.xlabel(r"$E_{\nu}$")
        logger.info(f"Saving to {savepath}")
        plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()

    def get_raw_pseudo_mc(self):
        data_dtype = np.dtype(
            [
                ("logE", float),
                ("trueE", float),
                ("sinDec", float),
                ("trueDec", float),
                ("ow", float),
                ("a_eff", float),
                ("sigma", float),
            ]
        )

        pseudo_mc = []

        with open(self.a_eff_path, "r") as f:
            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):
                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    true_e = 0.5 * (row[0] + row[1])
                    log_e = np.log10(true_e)
                    cos_zen = 0.5 * (row[2] + row[3])
                    zen = np.arccos(cos_zen)
                    true_dec = zen - np.pi / 2.0
                    sin_dec = np.sin(true_dec)
                    a_eff = row[4]

                    entry = tuple(
                        [log_e, true_e, sin_dec, true_dec, a_eff, a_eff, np.nan]
                    )

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

        # exp = exp[exp["sinDec"] < 0.]

        # pseudo_mc = pseudo_mc[]

        for i, x in enumerate([-5.0]):  # , -15.]):
            label = ["Upgoing", "Downgoing"][i]
            cut_value = np.sin(np.deg2rad(x))

            sign = np.sign(i - 0.5)

            # Cut down then up

            exp_cut = exp[(sign * exp["sinDec"]) > (sign * cut_value)]

            pseudo_mc_cut = pseudo_mc[(sign * pseudo_mc["sinDec"]) > (sign * cut_value)]

            log_e_exp = exp_cut["logE"]
            log_e_exp[log_e_exp < min(pseudo_mc_cut["logE"])] = min(
                pseudo_mc_cut["logE"]
            )

            # spread = np.linspace(-1., 1., 10)
            # weights = scipy.stats.norm.pdf(spread, scale=0.3)
            # print(log_e_vals[:,] * spread.T)

            # log_e_vals = np.dot(log_e_vals[:,None], spread[:,None].T).ravel()
            # weights = np.dot(pseudo_mc_cut["ow"][:, None],
            #                     weights[:,None].T).ravel()
            #
            # true_e = np.dot(pseudo_mc_cut["ow"][:, None],
            #                 np.ones_like(spread)[:,None].T).ravel()
            # print(log_e_vals)
            #
            # print("Weights", weights)
            # input("?")

            index = [3.7, 3.0][i]

            plt.figure()
            ax1 = plt.subplot(311)
            res = ax1.hist(log_e_exp, density=True)
            ax1.set_title("Energy Proxy (Data)")

            exp_vals = res[0]
            exp_bins = res[1]
            ax1.set_yscale("log")
            ax2 = plt.subplot(312, sharex=ax1)
            res = ax2.hist(
                pseudo_mc_cut["logE"],
                weights=pseudo_mc_cut["ow"] * pseudo_mc_cut["trueE"] ** -index,
                density=True,
                bins=exp_bins,
            )
            mc_vals = res[0]

            ax2.set_yscale("log")
            ax2.set_title(r"Expected True Energy ($E^{-" + str(index) + r"}$)")

            # Maps ratio of expected neutrino energies to energy proxy values
            # This can tell us about how true energy maps to energy proxy

            centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])

            # Fill in empty bins

            mc_vals = np.array(mc_vals)

            x = [-5.0] + list(centers) + [15.0]
            y = exp_vals / mc_vals
            y = [y[0]] + list(y) + [y[-1]]

            log_e_weighting = scipy.interpolate.interp1d(x, np.log(y))

            ax3 = plt.subplot(313)
            plt.plot(centers, exp_vals / mc_vals)
            plt.plot(centers, np.exp(log_e_weighting(centers)), linestyle=":")
            ax3.set_yscale("log")
            ax3.set_title("Ratio")
            ax3.set_xlabel(r"$\log_{10}(E)$")

            plt.tight_layout()

            save_path = energy_proxy_plot_path(self)

            try:
                os.makedirs(os.path.dirname(save_path))
            except OSError:
                pass

            save_path = os.path.dirname(save_path) + "/{0}-{1}.pdf".format(
                self.season_name, label
            )

            print("Saving to", save_path)

            plt.savefig(save_path)

            if show:
                plt.show()
            else:
                plt.close()

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

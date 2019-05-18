import os
import numpy as np
import csv
import pickle
from flarestack.shared import public_dataset_dir
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.utils.dataset_loader import data_loader
from flarestack.utils.make_SoB_splines import make_individual_spline_set
from flarestack.shared import SoB_spline_path, dataset_plot_dir
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

src_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

data_dir = src_dir + "raw_data/"
output_base_dir = public_dataset_dir + "all_sky_3_year/"
output_data_dir = output_base_dir + "events/"
pseudo_mc_dir = output_data_dir + "pseudo_mc/"

for path in [output_data_dir, pseudo_mc_dir]:

    try:
        os.makedirs(path)
    except OSError:
        pass


def data_path(season):
    return output_data_dir + season + ".npy"


def pseudo_mc_path(season):
    return pseudo_mc_dir + season + ".npy"


data_dtype = np.dtype([
    ('ra', np.float),
    ('dec', np.float),
    ('logE', np.float),
    ('sigma', np.float),
    ('time', np.float),
    ('sinDec', np.float)
])


datasets = ["IC79-2010", "IC86-2011", "IC86-2012"]


def parse_numpy_dataset():
    """Function to parse the .txt file  of events into a numpy format
    readable by flarestack, which is the saved in the products/ subdirectory.
    """

    for dataset in datasets:

        data = []

        path = data_dir + dataset + "-events.txt"

        with open(path, "r") as f:

            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):
                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    entry = (np.deg2rad(row[3]), np.deg2rad(row[4]),
                             row[1], np.deg2rad(row[2]),
                             row[0], np.sin(np.deg2rad(row[4]))
                             )

                    data.append(entry)

        data = np.array(data, dtype=data_dtype)

        exp_path = data_path(dataset)

        with open(exp_path, "wb") as f:
            print("Saving converted numpy array to", exp_path)
            pickle.dump(data, f)


def make_season(season_name):
    season_dict = {
        "Data Sample": "all_sky_3_year",
        "sinDec bins": np.unique(np.concatenate([
            np.linspace(-1., -0.9, 1 + 1),
            np.linspace(-0.9, -0.2, 4 + 1),
            np.linspace(-0.2, 0.2, 7 + 1),
            np.linspace(0.2, 0.9, 6 + 1),
            np.linspace(0.9, 1., 1 + 1),
        ])),
        "MJD Time Key": "time",
        "Name": season_name,
        "exp_path": data_path(season_name),
        "mc_path": None,
        "grl_path": None,
        "pseudo_mc_path": pseudo_mc_path(season_name)
    }
    return season_dict


ps_3_year = [make_season(x) for x in datasets]


# if __name__=="__main__":
#     parse_numpy_dataset()

def parse_effective_areas():
    """Function to parse effective areas .txt into a format that flarestack
    can use to build Signal/Background splines.
    """
    data_dtype = np.dtype([
        ('logE', np.float),
        ('trueE', np.float),
        ('sinDec', np.float),
        ('ow', np.float),
        ("sigma", np.float)
    ])

    for dataset in datasets:

        pseudo_mc = []

        path = data_dir + dataset + "-TabulatedAeff.txt"

        exp = data_loader(data_path(dataset))
        lower_e = min(exp["logE"])
        upper_e = max(exp["logE"])

        # Select only upgoing muons. For these events, the dominant
        # background is atmospheric neutrinos with a known spectrum of E^-3.7.
        # Downgoing events, on the other hand, are contaminated by sneaking
        # muon bundles which are harder to model.

        exp = exp[exp["sinDec"] > 0.]

        with open(path, "r") as f:

            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):

                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    true_e = 0.5*(row[0] + row[1])
                    log_e = np.log10(true_e)
                    sin_dec = -0.5*(row[2] + row[3])
                    a_eff = row[4]

                    randoms = [log_e]

                    for log_e in randoms:

                        # if log_e < 3.:
                        #     factor = 1e-4
                        # else:
                        #     factor = 1.
                        factor = 1.

                        entry = tuple([
                            log_e, true_e, sin_dec, a_eff*factor, np.nan
                        ])

                        pseudo_mc.append(entry)

        pseudo_mc = np.array(pseudo_mc, dtype=data_dtype)

        plt.figure()
        ax1 = plt.subplot(311)
        res = ax1.hist(exp["logE"], density=True)

        exp_vals = res[0]
        exp_bins = res[1]
        ax1.set_yscale("log")
        ax2 = plt.subplot(312, sharex=ax1)
        res = ax2.hist(
            pseudo_mc["logE"],
            weights=pseudo_mc["ow"] * pseudo_mc["trueE"]**-3.7,
            density=True, bins=exp_bins)
        mc_vals = res[0]
        ax2.set_yscale("log")

        # Maps ratio of expected neutrino energies to energy proxy values
        # This can tell us about how true energy maps to energy proxy

        centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])

        x = [-5.0] + list(centers) + [15.0]
        y = exp_vals / mc_vals
        y = [y[0]] + list(y) + [y[-1]]

        log_e_weighting = interp1d(x, np.log(y))

        ax3 = plt.subplot(313)
        plt.plot(centers, exp_vals/mc_vals)
        plt.plot(centers, np.exp(log_e_weighting(centers)),
                 linestyle=":")
        ax3.set_yscale("log")

        save_path = dataset_plot_dir + "/energy_proxy/all_sky_3_year/" \
                                       "{0}.pdf".format(dataset)

        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError:
            pass

        plt.savefig(save_path)
        plt.close()

        pseudo_mc["ow"] *= np.exp(log_e_weighting(pseudo_mc["logE"]))

        mc_path = pseudo_mc_path(dataset)

        with open(mc_path, "wb") as f:
            print("Saving converted numpy array to", mc_path)
            pickle.dump(pseudo_mc, f)


from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.utils.dataset_loader import data_loader

# mc = data_loader(ps_7year[0]["mc_path"])
# print(mc.dtype.names)
# for x in mc:
#     true_e = x["trueE"]
#     print(true_e, np.log10(true_e), x["logE"])
#     input("prompt")

parse_effective_areas()

for season in ps_3_year:
    # exp = data_loader(season["exp_path"])
    # mc = data_loader(season["pseudo_mc"])
    make_individual_spline_set(season, SoB_spline_path(season))
from flarestack.shared import catalogue_dir, limits_dir
import os
import pickle as Pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
tde_dir = os.path.abspath(os.path.dirname(__file__))
tde_cat_dir = tde_dir + "/tde_catalogues/"

individual_tdes = [
    "Swift J1644+57",
    "Swift J2058+05",
    "ASASSN-14li",
    "XMMSL1 J0740-85",
    "AT2018cow"
]


def individual_tde_cat(name):
    """Maps the name of a given TDE to the path of a source catalogue which
    contains only that TDE.

    :param name: Name of TDE
    :return: Path to catalogue
    """
    return tde_cat_dir + "individual_TDEs/" + name + "_catalogue.npy"


tde_catalogues = [
    "jetted",
    "gold",
    "silver",
    "obscured"
]


def tde_catalogue_name(name):
    """Maps the name of a given TDE catalogue to the path of a source
    catalogue which contains those TDEs in one combined catalogue.

    :param name: Name of TDE catalogue
    :return: Path to catalogue
    """

    return tde_cat_dir + "TDE_" + name + "_catalogue.npy"


def tde_cat_limit(name, index):
    path = limits_dir + "analyses/tde/compare_spectral_indices/Emin=100/" +\
        name + "/fit_weights/real_unblind/limit.pkl"

    if not os.path.isfile(path):
        raise Exception("{0} file not found".format(path))

    with open(path, "r") as f:
        data = Pickle.load(f)

    f = interp1d(data["x"], data["energy"])

    return f(index)


def make_limit_plot(name):
    path = limits_dir + "analyses/tde/compare_spectral_indices/Emin=100/" +\
        name + "/fit_weights/real_unblind/limit.pkl"

    if os.path.isfile(path):

        with open(path, "r") as f:
            data = Pickle.load(f)

        savepath = os.path.dirname(path) + "/" + name + "_limit_plot.pdf"

        f = interp1d(data["x"], np.log(data["energy"]))

        x_range = np.linspace(data["x"][0], data["x"][-1], 10)

        plt.figure()
        plt.errorbar(x_range, np.exp(f(x_range)),
                     yerr=.25 * np.exp(f(x_range)),
                     uplims=True)
        plt.plot(x_range, np.exp(f(x_range)))
        plt.ylabel(r"Isotropic-Equivalent $E_{\nu}$ (erg)")
        plt.xlabel(r"Spectral Index ($\gamma$)")
        plt.yscale("log")
        plt.title("Per-source limits for {0} TDEs".format(name))
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()


for cat in tde_catalogues:
    make_limit_plot(cat)
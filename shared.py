import os
import numpy as np
from config import scratch_path

# ==============================================================================
# Directory substructure creation
# ==============================================================================

# fs_dir is the path of the

fs_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

fs_scratch_dir = scratch_path + "flarestack__data/"

input_dir = fs_scratch_dir + "input/"
storage_dir = fs_scratch_dir + "storage/"
output_dir = fs_scratch_dir + "output/"
log_dir = fs_scratch_dir + "logs/"

catalogue_dir = input_dir + "catalogues/"

pickle_dir = storage_dir + "pickles/"

plots_dir = output_dir + "plots/"

acc_f_dir = input_dir + "acceptance_functions/"


gamma_range = [1., 4.]


def name_pickle_output_dir(name):
    return pickle_dir + name


def plot_output_dir(name):
    return plots_dir + name


def acceptance_path(season_name):
    return acc_f_dir + season_name + ".pkl"


def fit_setup(llh_kwargs, sources):

    # The default value for n_s is 1. It can be between 0 and 1000.
    p0 = [1.]
    bounds = [(0, 1000.)]
    names = ["n_s"]

    # If weights are to be fitted, then each source has an independent
    # n_s in the same 0-1000 range.
    if "Fit Weights?" in llh_kwargs.keys():
        if llh_kwargs["Fit Weights?"]:
            p0 = [1. for x in sources]
            bounds = [(0, 1000.) for x in sources]
            names = ["n_s (" + x["Name"] + ")" for x in sources]

    # If gamma is to be included as a fit parameter, then its default
    # value if 2, and it can range between 1 and 4.
    if "Fit Gamma?" in llh_kwargs.keys():
        if llh_kwargs["Fit Gamma?"]:
            p0.append(2.)
            bounds.append(tuple(gamma_range))
            names.append("Gamma")

    return p0, bounds, names


def k_to_flux(k):
    """k is a flux scale, with k=1 equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2.
    The k values can be converted into values for the flux.

    :param k: Flux scale
    :return: Flux value
    """
    return k * 10 ** -9

import os
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


def name_pickle_output_dir(name):
    return pickle_dir + name


def plot_output_dir(name):
    return plots_dir + name


def k_to_flux(k):

    # k is a flux scale, equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2

    return k * 10 ** -9

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
import os
import numpy as np
from . import config
import socket
import pickle as Pickle
from flarestack.core.energy_PDFs import gamma_range, EnergyPDF
import json

# ==============================================================================
# Directory substructure creation
# ==============================================================================

# fs_dir is the path of the

fs_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

fs_scratch_dir = config.scratch_path + "flarestack__data/"

input_dir = fs_scratch_dir + "input/"
storage_dir = fs_scratch_dir + "storage/"
output_dir = fs_scratch_dir + "output/"
cluster_dir = fs_scratch_dir + "cluster/"
log_dir = cluster_dir + "logs/"

cache_dir = storage_dir + "cache/"
cat_cache_dir = cache_dir + "catalogue_cache/"

catalogue_dir = input_dir + "catalogues/"
transients_dir = catalogue_dir + "transients/"
analysis_dir = input_dir + "analysis/"

pickle_dir = storage_dir + "pickles/"
inj_param_dir = pickle_dir + "injection_values/"

plots_dir = output_dir + "plots/"
limits_dir = output_dir + "limits/"
dataset_plot_dir = output_dir + "dataset_plots/"

illustration_dir = plots_dir + "illustrations/"

acc_f_dir = input_dir + "acceptance_functions/"
SoB_spline_dir = input_dir + "SoB_splines/"
bkg_spline_dir = input_dir + "bkg_splines/"
pc_dir = input_dir + "pull_corrections/"
pull_dir = pc_dir + "pulls/"
floor_dir = pc_dir + "floors/"

all_dirs = [
    fs_scratch_dir, input_dir, storage_dir, output_dir, cluster_dir, pc_dir,
    log_dir, catalogue_dir, acc_f_dir, pickle_dir, plots_dir,
    SoB_spline_dir, analysis_dir, illustration_dir, transients_dir,
    bkg_spline_dir, dataset_plot_dir, limits_dir, pull_dir, floor_dir,
    cache_dir, cat_cache_dir
]

# ==============================================================================
# Check host and specify path to dataset storage
# ==============================================================================

host = socket.gethostname()

if np.logical_or("ifh.de" in host, "zeuthen.desy.de" in host):
    dataset_dir = "/lustre/fs22/group/icecube/data_mirror/"
    skylab_ref_dir = dataset_dir + "mirror-7year-PS-sens/"
    print("Loading datasets from", dataset_dir, "(DESY)")
elif "icecube.wisc.edu" in host:
    dataset_dir = "/data/ana/analyses/"
    skylab_ref_dir = "/data/user/steinrob/mirror-7year-PS-sens/"
    print("Loading datasets from", dataset_dir, "(WIPAC)")
else:
    dataset_dir = "a dataset directory, which can be set with " \
                  "flarestack.shared.set_dataset_directory"


# Dataset directory can be changed if needed

def set_dataset_directory(path):
    """Sets the dataset directory to be a custom path, and exports this.

    :param path: Path to datasets
    """
    if not os.path.isdir(path):
        raise Exception("Attempting to set invalid path for datasets. "
                        "Directory", path, "does not exist!")
    print("Loading datasets from", path)

    global dataset_dir
    dataset_dir = path


# gamma_range = [1., 4.]
gamma_precision = .025

# Sets the minimum angular error

min_angular_err = np.deg2rad(0.0)

# Sets an angular error floor based on the 25th quantile

base_floor_quantile = 0.25


def floor_pickle(floor_dict):
    hash_dict = dict(floor_dict)
    hash_dict["season"] = dict(floor_dict["season"])
    try:
        del hash_dict["pull_name"]
    except KeyError:
        pass
    hash_dict["season"]["sinDec bins"] = list(hash_dict["season"]["sinDec bins"])
    unique_key = hash(json.dumps(hash_dict, sort_keys=True))
    return floor_dir + str(unique_key) + ".pkl"


def pull_pickle(pull_dict):
    hash_dict = dict(pull_dict)
    hash_dict["season"] = dict(pull_dict["season"])
    hash_dict["season"]["sinDec bins"] = list(hash_dict["season"]["sinDec bins"])
    unique_key = hash(json.dumps(hash_dict, sort_keys=True))
    return pull_dir + str(unique_key) + ".pkl"


def llh_energy_hash_pickles(llh_dict, season_dict):
    hash_dict = dict(llh_dict["llh_energy_pdf"])
    hash_dict["llh_name"] = llh_dict["llh_name"]
    key = hash(json.dumps(hash_dict, sort_keys=True))
    season_path = str(key) + "/" + season_dict["Name"] + "/" + \
                  season_dict["Data Sample"] + ".pkl"
    SoB_path = SoB_spline_dir + season_path
    acc_path = acc_f_dir + season_path
    return SoB_path, acc_path


band_mask_chunk_size = 500


def band_mask_hash_dir(catalogue):
    return cat_cache_dir + str(hash(str(catalogue))) + "/"


def band_mask_cache_name(season_dict, catalogue):
    n_chunks = int((len(catalogue) + band_mask_chunk_size - 1) \
               / band_mask_chunk_size)
    print("Breaking catalogue into", n_chunks, "chunks of", band_mask_chunk_size)

    cats = []

    for i in range(n_chunks):
        cat = catalogue[(i*band_mask_chunk_size):((i+1) * band_mask_chunk_size)]
        cats.append(cat)

    paths = [band_mask_hash_dir(cat) + season_dict["Data Sample"] + "/" +
             season_dict["Name"] + ".npz" for cat in cats]

    return cats, paths

def name_pickle_output_dir(name):
    return pickle_dir + name


def inj_dir_name(name):
    return inj_param_dir + name


def plot_output_dir(name):
    return plots_dir + name


def limit_output_path(name):
    path = limits_dir + name + "limit.pkl"
    return path


def acceptance_path(season):
    return acc_f_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def SoB_spline_path(season):
    return SoB_spline_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def bkg_spline_path(season):
    return bkg_spline_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def fit_setup(llh_kwargs, sources, fit_energy, flare=False):
    # The default value for n_s is 1. It can be between 0 and 10000.
    p0 = [1.]

    bounds = [(0.0, 1000.)]
    names = ["n_s"]

    # if "Fit Negative n_s?" in llh_kwargs.keys():
    #     if llh_kwargs["Fit Negative n_s?"]:
    #         bounds = [(-100., 1000.)]

    # If weights are to be fitted, then each source has an independent
    # n_s in the same 0-1000 range.
    if "Fit Weights?" in list(llh_kwargs.keys()):
        if llh_kwargs["Fit Weights?"]:
            p0 = [1. for x in sources]
            bounds = [bounds[0] for x in sources]
            names = ["n_s (" + x["Name"] + ")" for x in sources]

    if fit_energy:
        e_pdf = EnergyPDF.create(llh_kwargs["LLH Energy PDF"])
        e_seed, e_bounds, e_names = e_pdf.return_energy_parameters()
        p0 += e_seed
        bounds += e_bounds
        names += e_names

    if flare:
        names += ["Flare Start", "Flare End", "Flare Length"]

    return p0, bounds, names


k_flux_factor = 10 ** -9


def k_to_flux(k):
    """k is a flux scale, with k=1 equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2.
    The k values can be converted into values for the flux.

    :param k: Flux scale
    :return: Flux value
    """
    return k * k_flux_factor


def flux_to_k(flux):
    """k is a flux scale, with k=1 equal to (10)^-9 x (Gev)^-1 (s)^-1 (cm)^-2.
    The flux values can be converted into values for k.

    :param flux: Flux value
    :return: Flux scale (k)
    """
    return flux / k_flux_factor


def scale_shortener(scale):
    """Function to trim number of significant figures for flux scales when
    required for dictionary keys or saving pickle files.

    :param scale: Flux Scale
    :return: Flux Scale to 4.s.f
    """
    return '{0:.4G}'.format(float(scale))


def analysis_pickle_path(name):
    """Converts a unique Minimisation Handler name to a corresponding analysis
    config pickle. This pickle can be used to run a Minimisation Handler.

    :param name: unique Minimisation Handler name
    :return: Path to analysis pickle
    """
    analysis_path = analysis_dir + name

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = analysis_path + "dict.pkl"
    return pkl_file


def make_analysis_pickle(mh_dict):
    """Takes a Minimisation Handler Dictionary, finds the corresponding
    analysis pickle path, and saves the dictionary to this path

    :param mh_dict: Minimisation Handler dictionary
    """
    name = mh_dict["name"]

    pkl_file = analysis_pickle_path(name)

    with open(pkl_file, "wb") as f:
        Pickle.dump(mh_dict, f)

    return pkl_file


def weighted_quantile(values, quantiles, weight):
    """Calculated quantiles accounting for weights.

    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param weight: array-like of the same length as `array`
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    sample_weight = np.array(weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
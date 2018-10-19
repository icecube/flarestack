import os
import numpy as np
import config
import socket

# ==============================================================================
# Directory substructure creation
# ==============================================================================

# fs_dir is the path of the

fs_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

fs_scratch_dir = config.scratch_path + "flarestack__data/"

input_dir = fs_scratch_dir + "input/"
storage_dir = fs_scratch_dir + "storage/"
output_dir = fs_scratch_dir + "output/"
log_dir = fs_scratch_dir + "logs/"

catalogue_dir = input_dir + "catalogues/"
transients_dir = catalogue_dir + "transients/"
analysis_dir = input_dir + "analysis/"

pickle_dir = storage_dir + "pickles/"
inj_param_dir = pickle_dir + "injection_values/"

plots_dir = output_dir + "plots/"

illustration_dir = plots_dir + "illustrations/"

acc_f_dir = input_dir + "acceptance_functions/"
SoB_spline_dir = input_dir + "SoB_splines/"
bkg_spline_dir = input_dir + "bkg_splines/"

# ==============================================================================
# Check host and specify path to dataset storage
# ==============================================================================

host = socket.gethostname()

if "ifh.de" in host:
    dataset_dir = "/lustre/fs22/group/icecube/data_mirror/"
    skylab_ref_dir = dataset_dir + "mirror-7year-PS-sens/"
    print "Loading datasets from", dataset_dir, "(DESY)"
elif "icecube.wisc.edu" in host:
    dataset_dir = "/data/ana/analyses/"
    skylab_ref_dir = "/data/user/steinrob/mirror-7year-PS-sens/"
    print "Loading datasets from", dataset_dir, "(WIPAC)"
else:
    pass


# Dataset directory can be changed if needed

def set_dataset_directory(path):
    """Sets the dataset directory to be a custom path, and exports this.

    :param path: Path to datasets
    """
    if not os.path.isdir(path):
        raise Exception("Attempting to set invalid path for datasets. "
                        "Directory", path, "does not exist!")
    print "Loading datasets from", path

    global dataset_dir
    dataset_dir = path


gamma_range = [1., 4.]
gamma_precision = .025


def name_pickle_output_dir(name):
    return pickle_dir + name


def inj_dir_name(name):
    return inj_param_dir + name


def plot_output_dir(name):
    return plots_dir + name


def acceptance_path(season):
    return acc_f_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def SoB_spline_path(season):
    return SoB_spline_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def bkg_spline_path(season):
    return bkg_spline_dir + season["Data Sample"] + "/" + \
           season["Name"] + '.pkl'


def fit_setup(llh_kwargs, sources, flare=False):
    # The default value for n_s is 1. It can be between 0 and 1000.
    p0 = [1.]

    bounds = [(0.0, 1000.)]
    names = ["n_s"]

    # if "Fit Negative n_s?" in llh_kwargs.keys():
    #     if llh_kwargs["Fit Negative n_s?"]:
    #         bounds = [(-100., 1000.)]

    # If weights are to be fitted, then each source has an independent
    # n_s in the same 0-1000 range.
    if "Fit Weights?" in llh_kwargs.keys():
        if llh_kwargs["Fit Weights?"]:
            p0 = [1. for x in sources]
            bounds = [bounds[0] for x in sources]
            names = ["n_s (" + x["Name"] + ")" for x in sources]

    # If gamma is to be included as a fit parameter, then its default
    # value if 2, and it can range between 1 and 4.
    if "Fit Gamma?" in llh_kwargs.keys():
        if llh_kwargs["Fit Gamma?"]:
            p0.append(2.)
            bounds.append(tuple(gamma_range))
            names.append("Gamma")

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
    return '{0:.4G}'.format(scale)

import numpy as np
import logging
import os
import sys
import pandas as pd
import json
import copy
from tqdm import tqdm
import pickle
from zipfile import ZipFile
import matplotlib.pyplot as plt
import argparse
import shutil
import pylatex
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import (
    sim_dataset_dir,
    catalogue_dir,
    plot_output_dir,
    storage_dir,
    fs_scratch_dir,
)
from flarestack.utils.prepare_catalogue import custom_sources, cat_dtype
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.skylab_crosscheck.skylab_results import (
    crosscheck_with_same_scrambles,
)
from flarestack.core.angular_error_modifier import BaseAngularErrorModifier
from flarestack.core.llh import LLH
from flarestack.core.minimisation import MinimisationHandler


logger = logging.getLogger(__name__)


# set seed to make results reproduceable
np.random.seed(666)

# backup key
# set this variable in environment to enable storage of scrambles in that directory
# huge file size!
backup_env_key = "FLARESTACK_BACKUP_DIR"

# -----   source parameters ------ #

# number of sources
max_nsources = 10

# spectral index
gamma = 2.0

hemispheres = ["northern", "southern", "both"]

# sin(declinations) for northern and southern sources
sindecs = {
    "northern": list(np.linspace(0, 0.75, max_nsources, endpoint=True)),
    "southern": list(np.linspace(-0.5, 0, max_nsources, endpoint=False)),
}

# right ascensions
ras = {
    "northern": list(np.linspace(0, 2 * np.pi, max_nsources, endpoint=False)),
    "southern": list(np.linspace(0, 2 * np.pi, max_nsources, endpoint=False)),
}

# create an entry with northern and southern sources mixed
for d in [sindecs, ras]:
    d["both"] = list()
    for i in range(int(max_nsources / 2)):
        d["both"].append(d["northern"][i])
        d["both"].append(d["southern"][i])

# set up the injection dictionary
inj_dict = {
    # we will inject assuming a source that in constant over time
    "injection_sig_time_pdf": {"time_pdf_name": "steady"},
    # assuming a power law source
    "injection_energy_pdf": {"energy_pdf_name": "power_law", "gamma": gamma},
    # in order to determine the exact injected number of neutrinos we won't use poisson smear
    "poisson_smear_bool": False,
}


# --------    minimizer and likelihood parameters     ---------- #
# TODO: this should all be set after __name__ == '__main__' to avoid shadowing and
#  avoid confusing output directory structure!
minimizer_name = "large_catalogue"
dang = 10.0
likelihood_name = "standard_matrix"
llh_energy = {"energy_pdf_name": "power_law"}
llh_time = {"time_pdf_name": "steady"}
unblind_llh = {
    "llh_name": likelihood_name,
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "spatial_box_width": dang,
}
unblind_llh_spatial = {
    "llh_name": "spatial",
    "llh_sig_time_pdf": llh_time,
    "spatial_box_width": dang,
}


# -----------------    directories     --------------------- #
rawraw = "analyses/skylab_crosscheck/crosscheck_using_identical_scramble"
raw = f"{rawraw}/{minimizer_name}/dang{dang}"
data_dir = f"{sim_dataset_dir}{rawraw}"
this_cat_dir = f"{catalogue_dir}{rawraw}"
plot_dir = plot_output_dir(raw)
skylab_llh_scan_str = "_skylab_contour_scan"
zip_temp_dir = f"{storage_dir}/ziptemp"


# -----------------   pdf values   -------------------------- #
# get pdf values for first scramble
scramble_indice = 0


# =======================================    creating data scramble     =========================================== #

n_scrambles = 100
n_signal_range = np.linspace(0, 10, 6, endpoint=True)


def make_scramble(
    dataset, season, sources_in_hemisphere, number_of_sources, **inj_kwargs
):
    # load the 7-yr PS tracks dataset
    logger.debug("loading data")
    data = dataset.get_single_season(season)

    # get the sources
    sources_file = get_sources(sources_in_hemisphere, number_of_sources)
    sources = np.load(sources_file)

    # create injector
    injector = data.make_injector(sources, **inj_kwargs)

    # make the scrambled data with injected signal
    # the number of injected signal will be given explicitly in inj_kwargs
    # the parameter "scale" won't be used then
    # scale=0 just skips the injection step
    n_inj = inj_kwargs.get("fixed_n", 0)
    scale = 1 if n_inj > 0 else 0
    scramble = injector.create_dataset(scale=scale)

    return scramble


def check_if_backup_is_set():
    if backup_env_key in os.environ:
        return True
    else:
        return False


def backup_filename(fn):
    """returns the path to the same file as 'fn' in the backup directory"""
    backup_dir = os.environ.get(backup_env_key, None)
    backup_fn = f"{backup_dir}/{fn.split(fs_scratch_dir)[1]}"
    return backup_fn


def check_for_file_in_backup(fn):
    bfn = backup_filename(fn)
    if os.path.isfile(bfn):
        return True
    else:
        return False


def copy_to_backup(fn):
    bfn = backup_filename(fn)
    d = os.path.dirname(bfn)
    if not os.path.isdir(d):
        os.makedirs(d)
    shutil.copy2(fn, bfn)


def copy_from_backup(fn):
    bfn = backup_filename(fn)
    d = os.path.dirname(fn)
    if not os.path.isdir(d):
        os.makedirs(d)
    shutil.copy2(bfn, fn)


def get_scrambles(
    dataset, season, n_signal, sources_in_hemisphere, number_of_sources, **kwargs
):
    # make filename for each scramble
    directory = (
        f"{data_dir}/{list(dataset.values())[0].sample_name}/"
        f"{season}/ns_{n_signal}/{number_of_sources}{sources_in_hemisphere}_sources"
    )
    if not os.path.isdir(directory):
        logger.debug(f"making directory {directory}")
        os.makedirs(directory)
    fns = [
        f"{directory}/{i:.0f}.npy" for i in kwargs.get("indices", range(n_scrambles))
    ]

    # check if the files exist already and if the exist in the backup
    exist = np.array([os.path.isfile(fn) for fn in fns])
    exist_in_backup = np.array([check_for_file_in_backup(fn) for fn in fns])

    if np.any((~exist) & (~exist_in_backup)):
        # if the files do not exist in neither the directory nor the backup directory, produce the scrambles
        logger.warning(
            f"At least one file not found!\n"
            f"About to make {len(fns)} new scrambles for seasons {season} and {n_signal} signal neutrinos "
            f"for {number_of_sources} sources in the {sources_in_hemisphere} hemisphere. \n"
            f"Remember that the SkyLab results then have to be calculated again!"
        )

        i = kwargs.get("check", "")
        while i not in ["y", "n"]:
            i = input("Continue? [y/n] ")

        if i == "y":
            inj_kwargs = inj_dict
            inj_kwargs["fixed_n"] = n_signal

            for fn in fns:
                scramble = make_scramble(
                    dataset,
                    season,
                    sources_in_hemisphere,
                    number_of_sources,
                    **inj_kwargs,
                )
                logger.debug(f"saving {fn}")
                np.save(fn, scramble)

        else:
            print("exiting")
            sys.exit()

    if not check_if_backup_is_set():
        logger.warning(
            f'Backup directory not set! \nRun "export {backup_env_key}=/path/to/backup/dir'
        )

    else:
        if np.any((~exist) & exist_in_backup):
            # get the files from the backup directory to the working directory
            nfiles = len(np.array(fns)[(~exist) & exist_in_backup])
            logger.info(f"copying {nfiles} from backup.")

            if (nfiles > 10) and not kwargs.get("dont_check_backup_copy", False):
                input("continue? [hit any key, friend] ")

            for fn in tqdm(
                np.array(fns)[(~exist) & exist_in_backup],
                desc="Copying files from backup.",
            ):
                copy_from_backup(fn)

        if np.any(exist & (~exist_in_backup)):
            # if the files are present in the working directory but not in the backup directory,
            # copy to backup directory
            for fn in tqdm(
                np.array(fns)[exist & (~exist_in_backup)],
                desc="Copying files to backup.",
            ):
                copy_to_backup(fn)

    # write the location of the directory to file
    # this is used by the skylab script to get the scrambles
    fn_loc = f"{data_dir}/data_location_for_skylab.txt"
    if not os.path.isfile(fn_loc):
        logger.debug(f"writing scramble file location to file {fn_loc}")
        with open(fn_loc, "w") as f:
            f.write(data_dir)

    return fns


def get_scrambled_datasets(
    seasons,
    n_signal,
    sources_in_hemisphere,
    number_of_sources,
    dataset=ps_v002_p01.make_copy(),
    **kwa,
):
    """
    return the dataset but with scrambled experimental data
    :param seasons: str, which seasons to include in the dataset
    :param n_signal: float, number of signal neutrinos per source in data
    :param sources_in_hemisphere: 'northern', 'southern', 'both', specifies in which hemisphere the injected sources
                                    should lie.
    :param number_of_sources: int
    :param dataset: Dataset
    :return: list of Datasets
    """
    selected_dataset = dataset.get_seasons(*seasons)
    datasets = [
        copy.deepcopy(selected_dataset) for _ in kwa.get("indices", range(n_scrambles))
    ]

    for season_name in selected_dataset.keys():
        logger.debug(f"getting scrambles for season {season_name}")
        scramble_files = get_scrambles(
            dataset,
            season_name,
            n_signal,
            sources_in_hemisphere,
            number_of_sources,
            **kwa,
        )

        for i in range(len(datasets)):
            datasets[i][season_name].exp_path = scramble_files[i]

    return datasets


# ========================================       creating sources      ============================================ #


def make_sources(hemisphere, number_of_sources):
    """makes flarestack sources array"""
    sds, rs = (
        np.array(sindecs[hemisphere])[:number_of_sources],
        np.array(ras[hemisphere])[:number_of_sources],
    )
    names = [f"sindec={sd:.2f}" for sd in sds]
    weight = distance = [1.0] * len(sds)
    sources = np.empty(shape=len(sds), dtype=cat_dtype)
    for i, (n, r, sd, w, dist) in enumerate(zip(names, rs, sds, weight, distance)):
        new_source = custom_sources(
            name=n,
            ra=np.rad2deg(r),
            dec=np.rad2deg(np.arcsin(sd)),
            weight=w,
            distance=dist,
        )
        sources[i] = new_source
    return sources


def make_sources_location_file():
    """dumps the location of the files containing the source positions to a file that will be used by SkyLab"""
    fn = f"{this_cat_dir}/sources_location_for_skylab.txt"
    if not os.path.isfile(fn):
        logger.debug(f"writing sources location to file {fn}")
        with open(fn, "w") as f:
            f.write(this_cat_dir)


def get_sources(hemisphere, number_of_sources):
    """
    get sources
    :param hemisphere: str, 'northern', 'southern', 'both'
    :param number_of_sources: int
    :return: source filename
    """
    fn = f"{this_cat_dir}/{hemisphere}_{number_of_sources:.0f}_sources.npy"

    make_sources_location_file()

    if not os.path.isfile(fn):
        logger.info(f"sources file {fn} not found. Making sources")
        sources = make_sources(hemisphere, number_of_sources)
        np.save(fn, sources)

    return fn


def plot_skymap(declinations, right_ascensions, fn, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(projection="hammer")
    ax.scatter(right_ascensions, declinations)
    ax.set_xlabel(kwargs.get("xlabel", "RA"))
    ax.set_ylabel(kwargs.get("ylabel", "DEC"))
    ax.set_title(kwargs.get("title"))
    ax.grid()
    fig.tight_layout()
    fig.savefig(fn)
    plt.close()


# ======================================    use skylab splines   ======================================== #


def get_skylab_splines(season, kernel, *args, **kwargs):
    """loads the splines produced by SkyLab with the specified kernel and season"""
    filename = "splines/{0}_kernel{1}.pkl".format(season, kernel)
    skylab_splines = get_skylab_results(filename)
    return skylab_splines


def get_skylab_spline_scramble_fn(
    season, kernel, n_signal, number_of_sources, hemisphere, gamma, scramble_ind
):
    filename = (
        f"splines/{season}_ns{n_signal}_{number_of_sources}sources_{hemisphere}hemisphere_gamma{gamma}_"
        f"scramble{scramble_ind}_kernel{kernel}.pkl"
    )
    return filename


def get_llh_dict_for_skylab_splines(llh_dict_in, kernel, **kwargs):
    """returns a modification of the given llh_dict to use SkyLab splines produced with te specified kernel"""
    logger.debug(f'old name is {llh_dict_in["llh_name"]}')
    llh_dict = copy.deepcopy(llh_dict_in)
    new_name = define_llh_subclass(llh_dict["llh_name"])
    llh_dict["llh_name"] = new_name
    llh_dict["kernel"] = kernel
    logger.debug(f"Adding kwargs {kwargs}")
    llh_dict["spline_dict"] = kwargs
    return llh_dict


def define_llh_subclass(likelihood_name):
    """defines a new LLH class that will use SkyLab splines"""

    new_llh_name = f"{likelihood_name}_with_skylab_splines"

    if new_llh_name not in LLH.subclasses:
        logger.info(f"registering {new_llh_name}")

        @LLH.register_subclass(new_llh_name)
        class SkyLabSplinesLLH(LLH.subclasses[likelihood_name]):
            def __init__(self, season, sources, llh_dict):
                super().__init__(season, sources, llh_dict)
                kernel = llh_dict["kernel"]
                # a dictionary can be passed under "spline_dict"
                kwargs = llh_dict["spline_dict"]
                logger.debug(f"kwargs are {kwargs}")
                self.SoB_spline_2Ds = get_skylab_splines(
                    season.season_name, kernel, **kwargs
                )
                logger.debug(f"loaded {len(self.SoB_spline_2Ds)} splines")

        MinimisationHandler.subclasses[minimizer_name].compatible_llh.append(
            new_llh_name
        )

    return new_llh_name


# ======================================    get pdf values   ======================================== #


def get_pdf_values(
    scramble_ind,
    season,
    n_signal,
    hemisphere,
    nsources,
    gamma,
    use_skylab_energy_splines=False,
    skylab_kernel=None,
):
    """
    Get the raw PDF values for one scrambled Datset
    :param scramble_ind: int, The indice of the scrambled dataset
    :param season: str, seasons name
    :param n_signal: int, number of injected signal neutrinos
    :param hemisphere: str, hemisphere
    :param nsources: int, number of sources
    :param gamma: float, spectral index of injection power law
    :param use_skylab_energy_splines: bool, whether to use splines produced by SkyLab
    :param skylab_kernel: the kernel with which the SkyLab splines were produced
    :return: dict with keys "spatial" and "energy"
    """

    llh_dict = copy.deepcopy(unblind_llh)

    if use_skylab_energy_splines:
        if isinstance(skylab_kernel, type(None)):
            raise ValueError("When using SkyLab splines, specify SkyLab kernel!")
        logger.debug("using skylab splines")
        kw = (
            {
                "n_signal": n_signal,
                "number_of_sources": nsources,
                "hemisphere": hemisphere,
                "gamma": gamma,
                "scramble_ind": scramble_ind,
            }
            if season == "IC86_1"
            else dict()
        )
        logger.debug(kw)
        llh_dict = get_llh_dict_for_skylab_splines(llh_dict, skylab_kernel, **kw)
        logger.debug(f"keys of llh_dict: {llh_dict.keys()}")

    scrambled_dataset = get_scrambled_datasets(
        [season], n_signal, hemisphere, nsources
    )[scramble_ind]
    sources = np.load(get_sources(hemisphere, nsources))
    return pdf_values(scrambled_dataset, sources, llh_dict, season)


def pdf_values(scrambled_dataset, sources, llh_dict, season):
    llh = LLH.create(scrambled_dataset[season], sources, llh_dict)
    pull_corrector = BaseAngularErrorModifier.create(
        scrambled_dataset[season],
        llh_dict["llh_energy_pdf"],
        llh_dict.get("floor_name", "static_floor"),
        llh_dict.get("pull_name", "no_pull"),
    )

    pdf_values_dict = {"spatial": [], "energy": []}

    # get the spatial PDF value
    def SoB_pdf(x):
        pdf_per_source = list()
        for source in sources:
            pdf_per_source.append(
                llh.signal_pdf(source, x) / llh.background_pdf(source, x)
            )
        logger.debug(
            f"type of pdf is {type(pdf_per_source)}, shape is {np.shape(pdf_per_source)}"
        )
        pdf = [max(i) for i in np.array(pdf_per_source).T]
        logger.debug(f"pdf is {np.array(pdf)[:5]}")
        return pdf

    scrambled_data = np.load(scrambled_dataset[season].exp_path)
    spatial_cache = pull_corrector.create_spatial_cache(scrambled_data, SoB_pdf)
    spatial = list(pull_corrector.estimate_spatial(gamma, spatial_cache))
    pdf_values_dict["spatial"] = spatial

    # get the energy PDF value
    energy_cache = llh.create_SoB_energy_cache(scrambled_data)
    energy = list(llh.estimate_energy_weights(gamma, energy_cache))
    pdf_values_dict["energy"] = energy

    return pdf_values_dict


# ===================================       minimizer     ================================================= #


def get_minimizer_dict(
    name, nsources, hemisphere, data, llh_dict=copy.deepcopy(unblind_llh)
):
    catalogue_path = get_sources(hemisphere, nsources)

    minimizer_dictionary = {
        "name": name,
        "mh_name": "fixed_weights"
        if "spatial" in name
        else minimizer_name,  # TODO: find a way to not do this!
        "dataset": data,
        "catalogue": catalogue_path,
        "llh_dict": llh_dict,
    }

    return minimizer_dictionary


# ========================================     get skylab results   ======================================= #


def get_skylab_results(file):
    """gets the specified file containing the skylab results and returns the contained dictionary"""
    skylab_result_file = crosscheck_with_same_scrambles(file)
    if os.path.isfile(skylab_result_file):
        logger.debug(f"loading {skylab_result_file}")

        # if file is zip compressed archive, extract in scratch directory
        if "zip" in skylab_result_file:
            logger.debug(f"extracting from {skylab_result_file}")
            with ZipFile(skylab_result_file, "r") as zipObj:
                # Extract all the contents of zip file in different directory
                zipObj.extractall(zip_temp_dir)
            skylab_result_file = f"{zip_temp_dir}/res.pkl"

        if skylab_result_file.endswith(".json"):
            with open(skylab_result_file, "r") as f:
                skylab_res = json.load(f)
        elif skylab_result_file.endswith(".pkl"):
            with open(skylab_result_file, "rb") as f:
                skylab_res = pickle.load(f, encoding="latin1")
        else:
            raise ValueError(f"Can't load file {skylab_result_file}")

    else:
        logger.warning(f"SkyLab result file {skylab_result_file} doesn't exist")
        skylab_res = dict()

    return skylab_res


def get_skylab_scratch_dir():
    env_var = "SKYLAB_SCRATCH_DIR"
    skylab_scratch = os.environ.get(env_var, None)
    if not skylab_scratch:
        logger.warning(
            f"No environment variable {env_var}! Please specify SkyLab Scratch Directory!"
        )
    return skylab_scratch


def skylab_llh_scan_filename(nsources, season_name, ns, index, hemisphere, kernel):
    skylab_scratch_dir = get_skylab_scratch_dir()

    if not skylab_scratch_dir:
        logger.warning("could not get filename!")
        return

    else:
        filename = (
            f"{skylab_scratch_dir}/plots/fs_crosscheck_with_same_scrambles/"
            f"{nsources}sources/kernel{kernel}/{season_name}/{hemisphere}hemisphere/ns{ns}/llh_scans/{index}.pdf"
        )
        if not os.path.isfile(filename):
            logger.warning(f"File {filename} doesn't exist!")
            return
        else:
            return filename


def copy_single_skylab_llh_scan(target_filename, **kwargs):
    fn = skylab_llh_scan_filename(**kwargs)
    if not fn:
        logger.warning(f"Couldn't copy SkyLab LLH scan for {kwargs}")
    else:
        logger.debug(f"copying {fn} to {target_filename}")
        shutil.copy2(fn, target_filename)


def copy_skylab_llh_scans(indices, target_directory, **kwargs):
    """copies SkyLab's LLH contour scans to Flarestack output directory for comparison"""
    if isinstance(indices, type(None)):
        logger.warning("No indices given! Can't copy SkyLab LLH scans!")
        return

    for i in indices:
        filename = f"{target_directory}/{i}{skylab_llh_scan_str}.pdf"
        copy_single_skylab_llh_scan(filename, index=i, **kwargs)


# ========================================   analyse results   ============================================ #


def get_outliers(params, skylab_params, **kwargs):
    """
    Returns the indices where the params and skylab_params differ the most
    :param params: dict, parameter names as keys and their values in lists as values
    :param skylab_params: same as params but for SkyLab results
    :param kwargs: if n_outliers is given, specifiy how many outlier indices should be given per parameter
    :return: list
    """

    logger.debug("getting outliers")

    dif = {
        param: abs(
            np.array(skylab_params.get(param, [np.nan] * n_scrambles))
            - np.array(params[param])
        )
        for param in params
        if ("pdf" not in param) and ("spatial" not in param)
    }

    includes_numbers = np.array(
        [np.any([~np.isnan(a) for a in dif[param]]) for param in dif]
    )
    if np.all(~includes_numbers):
        logger.warning("No SkyLab results! Couldn't get outlier!")
        return

    dif_sorted_indices = {param: np.argsort(-np.array(dif[param])) for param in dif}

    outlier_indices = list()
    for param in dif:
        for i in range(kwargs.get("n_outliers", 3)):
            ind = dif_sorted_indices[param][i]
            if ind not in outlier_indices:
                outlier_indices.append(ind)

    logger.debug(f"outliers: {outlier_indices}")

    return outlier_indices


def get_outliers_filename():
    return f"{storage_dir}/{raw}{minimizer_name}_outlier_indices.pkl"


def save_outlier_dict(d):
    fn = get_outliers_filename()
    with open(fn, "wb") as f:
        pickle.dump(d, f)


# ========================================        plots         =========================================== #

# ------------------------  pdf values plots --------------------------------------- #


def param_label(param_name):
    if "gamma" in param_name:
        return "$\gamma$"
    else:
        return param_name


def make_pdf_values_scatter_plots(flarestack_dict, skylab_dict, directory, **kwargs):
    filenames = list()
    captions = list()

    for i, (kind, flarestack_list) in enumerate(flarestack_dict.items()):
        captions.append(f"{kind} PDF")
        filename = f"{directory}/{kind}.pdf"
        filenames.append(filename)

        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={
                "width_ratios": [3, 1],
                "wspace": 0,
                "height_ratios": [1, 3],
                "hspace": 0,
            },
            figsize=(5.85, 5.85),
        )

        qs = [0.5, 0.05, 0.95]

        skylab_list = skylab_dict.get(kind, [np.nan] * len(flarestack_list))
        mask = (np.array(flarestack_list) <= 1e-30) & (np.array(skylab_list) <= 1e-30)
        perc_zero = len(np.where(mask)[0]) / len(mask) * 100

        # central scatter plot
        axs[1][0].scatter(
            np.array(skylab_list)[~mask],
            np.array(flarestack_list)[~mask],
            color="k",
            s=0.6,
            label=f"{kind} PDF values",
        )  # \n{perc_zero:.2f}% at 0')
        axs[1][0].set_xscale("log")
        axs[1][0].set_yscale("log")
        xlims, ylims = axs[1][0].get_xlim(), axs[1][0].get_ylim()
        lower_lim = min(
            min(np.array(flarestack_list)[np.array(flarestack_list) >= 1e-30]),
            min(np.array(skylab_list)[np.array(skylab_list) >= 1e-30]),
        )
        lim = (lower_lim, max(max(xlims), max(ylims)))
        logger.debug(f"axis limits are {lim}")

        # if (perc_zero == 0) & (lim[0] >= 10):
        #     lim = (min((min(flarestack_list), min(skylab_list))) / 10,
        #            max(lim))

        # plot 1:1 reference line
        plot_straight_line = [lim[0] - 1, lim[1] + 1]
        axs[1][0].plot(
            plot_straight_line,
            plot_straight_line,
            ls="--",
            color="k",
            alpha=0.7,
            linewidth=0.6,
            label="1:1 reference",
        )

        # set same limits to skylab and flarestack axis for better comparison
        axs[1][0].set_xlim(lim)
        axs[1][0].set_ylim(lim)
        axs[1][0].set_ylabel("Flarestack")
        axs[1][0].set_xlabel("SkyLab")
        axs[1][0].legend()

        try:
            # skylab histograms
            skylab_mask = np.array(skylab_list) <= 1e-30
            skylab_loglist = np.log10(np.array(skylab_list)[~skylab_mask])
            sl_quantiles = np.quantile(skylab_loglist, qs)
            axs[0][0].hist(skylab_loglist, color="k", alpha=0.5)
            axs[0][0].axvline(
                sl_quantiles[0], color="k", label=f"median {10 ** sl_quantiles[0]:.2f}"
            )
            axs[0][0].axvline(
                sl_quantiles[1],
                color="k",
                ls="--",
                label=rf"IC$_{{90}}$: [{10 ** sl_quantiles[1]:.2f}, {10 ** sl_quantiles[2]:.2f}]",
            )
            axs[0][0].axvline(sl_quantiles[2], color="k", ls="--")
            axs[0][0].set_xlim(np.log10(lim))
            axs[0][0].legend()
            axs[0][0].axis("off")

            # flarestack histograms
            flarestack_mask = np.array(flarestack_list) <= 1e-30
            flarestack_loglist = np.log10(np.array(flarestack_list)[~flarestack_mask])
            fs_quantiles = np.quantile(flarestack_loglist, qs)
            axs[1][1].hist(
                flarestack_loglist, color="k", alpha=0.5, orientation="horizontal"
            )
            axs[1][1].axhline(
                fs_quantiles[0], color="k", label=f"median {10 ** fs_quantiles[0]:.2f}"
            )
            axs[1][1].axhline(
                fs_quantiles[1],
                color="k",
                ls="--",
                label=rf"IC$_{{90}}$: [{10 ** fs_quantiles[1]:.2f}, {10 ** fs_quantiles[2]:.2f}]",
            )
            axs[1][1].axhline(fs_quantiles[2], color="k", ls="--")
            axs[1][1].set_ylim(np.log10(lim))
            axs[1][1].legend()
            axs[1][1].axis("off")

        except ValueError as e:
            raise Exception(
                f"Min(Skylab)={min(np.array(skylab_list)[~mask])}, "
                f"Min(Flarestack)={min(np.array(flarestack_list)[~mask])}\n"
                f"raised Error: {e}"
            )

        # disable unused subplot
        axs[0][1].axis("off")

        logger.debug(f"saving under {filename}")
        fig.savefig(filename)
        plt.close()

    return filenames, captions


def make_pdf_values_plots(
    flarestack_dict, skylab_dict, directory, scramble_ind, **kwargs
):
    if not skylab_dict:
        logger.warning(f"No data in SkyLab dictionary! Can't make plot.")
        return

    if not os.path.isdir(directory):
        logger.debug(f"making directory {directory}")
        os.makedirs(directory)

    scatter_fns, scatter_captions = make_pdf_values_scatter_plots(
        flarestack_dict, skylab_dict, directory
    )

    # create document
    filename = kwargs.get("filename", f"{directory}/summary")
    geometry_options = {"margin": "0.5in", "right": "2cm", "left": "2cm"}
    doc = pylatex.Document(
        filename, geometry_options=geometry_options, page_numbers=False
    )
    with doc.create(pylatex.Section(kwargs.get("title", ""), numbering=False)):
        with doc.create(pylatex.Figure(position="h!")):
            logger.debug("adding figure")
            width = 1 / len(scatter_fns) - 0.005
            for file, caption in zip(scatter_fns, scatter_captions):
                with doc.create(
                    pylatex.SubFigure(
                        position="b", width=pylatex.NoEscape(rf"{width:.2f}\linewidth")
                    )
                ) as sub:
                    sub.add_image(file)
                    sub.add_caption(caption)

            doc.append(pylatex.VerticalSpace("1cm"))

    doc.generate_pdf(filename, clean_tex=True, clean=True, silent=True)


# ----------------------------------  ts value plots  --------------------------------------- #


def make_scramble_line_plot(
    parameter_dict,
    skylab_parameter_dict,
    filename,
    injected_ns,
    injected_gamma,
    nscrambles=n_scrambles,
):
    """plots the SkyLab and Flarestack results against number of scramble in a line plot"""

    fig, axs = plt.subplots(
        len(parameter_dict.keys()),
        sharex="all",
        figsize=(5.85, 3.6154988341868854 / 2 * len(parameter_dict.keys())),
    )
    axs_dict = dict()
    fs_patch = sl_patch = None

    for i, (param_name, param_list) in enumerate(parameter_dict.items()):
        if "pdf" in param_name:
            continue

        skylab_param_list = skylab_parameter_dict.get(param_name, [np.nan] * nscrambles)

        if np.any(np.isnan(skylab_param_list)):
            logger.warning("NaN in SkyLab result list!")

        axs_dict[param_name] = axs[i]
        axs_dict[param_name].plot([], [], " ", label=param_name)
        fs_patch = axs_dict[param_name].plot(
            list(range(n_scrambles)), param_list, color="r"
        )
        sl_patch = axs_dict[param_name].plot(
            list(range(n_scrambles)), skylab_param_list, color="b", ls="--"
        )

    injected_patch = axs_dict["ns"].axhline(float(injected_ns), color="k", ls="--")
    axs_dict["gamma"].axhline(injected_gamma, color="k", ls="--")

    for param_name, ax in axs_dict.items():
        ax.set_ylabel(param_name)

    axs[-1].set_xlabel("scramble")

    fig.legend(
        [fs_patch[0], sl_patch[0], injected_patch],
        ["Flarestack", "SkyLab", "injected value"],
        loc="lower center",
        ncol=3,
    )
    fig.subplots_adjust(bottom=0.75)
    fig.tight_layout()
    logger.debug(f"saving figure under {filename}")
    fig.savefig(filename)
    plt.close()


def make_combined_scatter_plot(
    parameter_dict, skylab_parameter_dict, filename, nscrambles=n_scrambles, **kwargs
):
    """plots the Flaresatck results against the SkyLab results in a scatter plot. One figure with subplots for each
    parameter"""

    fig, axs = plt.subplots(
        len(parameter_dict.keys()),
        figsize=(5.85 / 2, 5.85 / 2 * len(parameter_dict.keys())),
    )
    data_patch = outlier_patch = ref_patch = None
    outlier_indices = get_outliers(parameter_dict, skylab_parameter_dict, **kwargs)

    for i, (param_name, param_list) in enumerate(parameter_dict.items()):
        if "pdf" in param_name:
            continue

        skylab_param_list = skylab_parameter_dict.get(param_name, [np.nan] * nscrambles)

        if np.any(np.isnan(skylab_param_list)):
            logger.warning("NaN in SkyLab result list!")

        ax = axs[i]
        data_patch = ax.scatter(skylab_param_list, param_list, color="k", s=0.6)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        if (not isinstance(outlier_indices, type(None))) and (len(outlier_indices) > 0):
            outlier_patch = ax.scatter(
                np.array(skylab_param_list)[outlier_indices],
                np.array(param_list)[outlier_indices],
                color="r",
                s=0.8,
            )
            for ind in outlier_indices:
                ax.text(
                    skylab_param_list[ind] + max(xlim) / 100,
                    param_list[ind],
                    s=f"{ind}",
                    color="r",
                    fontsize=4,
                )

        plot_straight_line = [xlim[0] - 1, xlim[1] + 1]
        ref_patch = ax.plot(
            plot_straight_line,
            plot_straight_line,
            ls="--",
            color="k",
            alpha=0.7,
            linewidth=0.6,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(rf"{param_label(param_name)}$_{{Flarestack}}$")
        ax.set_xlabel(rf"{param_label(param_name)}$_{{SkyLab}}$")

    fig.legend(
        [data_patch, outlier_patch, ref_patch[0]],
        ["data", "outliers", "1:1 reference"],
        loc="lower center",
        ncol=3,
    )
    fig.tight_layout()
    logger.debug(f"saving figure under {filename}")
    fig.savefig(filename)
    plt.close()


def make_individual_scatter_plots(
    parameter_dict,
    skylab_parameter_dict,
    directory,
    injected_ns,
    injected_gamma,
    nscrambles=n_scrambles,
    **kwargs,
):
    """Plots the Flarestack results against the SkyLab results and histogramms on the sides. One figure per parameter"""

    if not os.path.isdir(directory):
        os.makedirs(directory)

    outlier_indices = get_outliers(parameter_dict, skylab_parameter_dict)

    for i, (param_name, param_list) in enumerate(parameter_dict.items()):
        if "pdf" in param_name:
            continue

        logger.debug(f"making plot for {param_name}")

        injected_value = (
            injected_ns
            if "ns" in param_name
            else injected_gamma
            if param_name == "gamma"
            else np.nan
        )
        qs = [0.5, 0.05, 0.95]

        skylab_param_list = skylab_parameter_dict.get(param_name, [np.nan] * nscrambles)
        if np.any(np.isnan(skylab_param_list)):
            logger.warning(
                f"NaN in SkyLab result list! Keys in SkyLab dict: {skylab_parameter_dict.keys()}"
            )
            continue

        param_fig, param_axs = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={
                "width_ratios": [3, 1],
                "wspace": 0,
                "height_ratios": [1, 3],
                "hspace": 0,
            },
            figsize=(5.58, 5.58),
        )

        # central scatter plot
        param_axs[1][0].scatter(
            skylab_param_list, param_list, color="k", s=0.6, label="data"
        )
        xlim, ylim = param_axs[1][0].get_xlim(), param_axs[1][0].get_ylim()

        # plot outliers
        if (not isinstance(outlier_indices, type(None))) and (len(outlier_indices) > 0):
            param_axs[1][0].scatter(
                np.array(skylab_param_list)[outlier_indices],
                np.array(param_list)[outlier_indices],
                color="r",
                s=0.8,
            )
            for ind in outlier_indices:
                param_axs[1][0].text(
                    skylab_param_list[ind] + max(xlim) / 100,
                    param_list[ind],
                    s=f"{ind}",
                    color="r",
                    fontsize=4,
                )

        # get nice axis limits
        lower_lim = (
            0
            if param_name == "ns"
            else 1
            if param_name == "gamma"
            else min((min(xlim), min(ylim)))
        )
        upper_lim = 4 if param_name == "gamma" else max(max(xlim), max(ylim))
        lim = (lower_lim, upper_lim)

        # plot a 1:1 reference line
        plot_straight_line = [lim[0] - 1, lim[1] + 1]
        param_axs[1][0].plot(
            plot_straight_line,
            plot_straight_line,
            ls="--",
            color="k",
            alpha=0.7,
            linewidth=0.6,
            label="1:1 reference",
        )

        param_axs[1][0].set_xlim(lim)
        param_axs[1][0].set_ylim(lim)
        param_axs[1][0].axvline(
            injected_value, color="r", ls="--", label="injection value"
        )
        param_axs[1][0].axhline(injected_value, color="r", ls="--")
        param_axs[1][0].set_xlabel(rf"{param_label(param_name)}$_{{SkyLab}}$")
        param_axs[1][0].set_ylabel(rf"{param_label(param_name)}$_{{Flarestack}}$")
        param_axs[1][0].legend()

        # SkyLab histogram
        skylab_quantiles = np.quantile(skylab_param_list, qs)
        param_axs[0][0].hist(skylab_param_list, color="k", alpha=0.5)
        param_axs[0][0].axvline(
            skylab_quantiles[0], color="k", label=f"median {skylab_quantiles[0]:.2f}"
        )
        param_axs[0][0].axvline(
            skylab_quantiles[1],
            color="k",
            ls="--",
            label=rf"IC$_{{90}}$: "
            rf"[{skylab_quantiles[1]:.2f}, {skylab_quantiles[2]:.2f}]",
        )
        param_axs[0][0].axvline(skylab_quantiles[2], color="k", ls="--")
        param_axs[0][0].axvline(injected_value, color="r", ls="--")
        param_axs[0][0].set_xlim(lim)
        param_axs[0][0].legend()
        param_axs[0][0].axis("off")

        # Flaresatck histogram
        flarestack_quantiles = np.quantile(param_list, qs)
        param_axs[1][1].axhline(
            flarestack_quantiles[0],
            color="k",
            label=f"median {flarestack_quantiles[0]:.2f}",
        )
        param_axs[1][1].axhline(
            flarestack_quantiles[1],
            color="k",
            ls="--",
            label=rf"IC$_{{90}}$: "
            rf"[{flarestack_quantiles[1]:.2f}, {flarestack_quantiles[2]:.2f}]",
        )
        param_axs[1][1].axhline(flarestack_quantiles[2], color="k", ls="--")
        param_axs[1][1].hist(param_list, color="k", alpha=0.5, orientation="horizontal")
        param_axs[1][1].axhline(injected_value, color="r", ls="--")
        param_axs[1][1].set_ylim(lim)
        param_axs[1][1].legend()
        param_axs[1][1].axis("off")

        # disable unused aubplot
        param_axs[0][1].axis("off")

        param_axs[0][0].set_title(f"{kwargs.get('add_to_title', '')}{param_name}")
        param_fig.tight_layout()
        filename = f"{directory}/{param_name}.pdf"
        logger.debug(f"saving {filename}")
        param_fig.savefig(filename)
        plt.close()


def make_plots_comparing_smoothings(
    directory,
    flaresatck_smoothing_param_dict,
    skylab_smoothing_param_dict,
    skylab_param_dict,
    injected_ns,
    injected_gamma,
    nscrambles=n_scrambles,
):
    if not os.path.isdir(directory):
        os.makedirs(directory)

    outlier_indices = get_outliers(flaresatck_smoothing_param_dict, skylab_param_dict)

    for i, (param_name, fs_smoothing_param_list) in enumerate(
        flaresatck_smoothing_param_dict.items()
    ):
        if "pdf" in param_name:
            continue

        injected_value = (
            injected_ns
            if param_name == "ns"
            else injected_gamma
            if param_name == "gamma"
            else np.nan
        )
        injected_value = float(injected_value)

        skylab_param_list = skylab_param_dict.get(param_name, [np.nan] * nscrambles)
        if np.any(np.isnan(skylab_param_list)):
            logger.warning("NaN in SkyLab result list!")

        sl_smoothing_param_list = skylab_smoothing_param_dict[param_name]
        difference = np.array(sl_smoothing_param_list) - np.array(
            fs_smoothing_param_list
        )

        param_fig, param_axs = plt.subplots(figsize=(5.58, 5.58))

        label = "SkyLab smoothing - Flaresatck smoothing"
        for i, (y, dy, x) in enumerate(
            zip(fs_smoothing_param_list, difference, skylab_param_list)
        ):
            c = "k"
            if (not isinstance(outlier_indices, type(None))) and (i in outlier_indices):
                c = "r"
                param_axs.text(x, y, f"{i}", color="r", fontsize=4)

            param_axs.arrow(
                x,
                y,
                0,
                dy,
                length_includes_head=True,
                label=label,
                color=c,
                head_width=0.02,
                width=0.0002,
            )
            label = ""

        lower_lim = 0 if param_name != "gamma" else 0.9
        upper_lim = (
            max(fs_smoothing_param_list + sl_smoothing_param_list) + 0.1
            if param_name != "gamma"
            else 4.1
        )

        plot_straight_line = [lower_lim - 1, upper_lim + 1]
        param_axs.plot(
            plot_straight_line,
            plot_straight_line,
            ls="--",
            color="k",
            alpha=0.7,
            linewidth=0.6,
            label="1:1 reference",
        )

        param_axs.axvline(injected_value, color="r", ls="--", label="injection value")
        param_axs.axhline(injected_value, color="r", ls="--")

        param_axs.set_xlabel(rf"{param_label(param_name)}$_{{SkyLab}}$")
        param_axs.set_ylabel(rf"{param_label(param_name)}$_{{Flarestack}}$")
        param_axs.set_xlim((lower_lim, upper_lim))
        param_axs.set_ylim((lower_lim, upper_lim))
        param_axs.legend()

        param_fig.tight_layout()
        filename = f"{directory}/{param_name}.pdf"
        logger.debug(f"saving to {filename}")
        param_fig.savefig(filename)
        plt.close()


def make_llh_scans_of_outliers(
    directory,
    seasons,
    n_injected,
    hemisphere,
    number_of_sources,
    dataset,
    indices,
    use_skylab_splines=False,
    skylab_kernel=None,
):
    if not indices:
        logger.warning("No indices given! Can't make LLH scans!")
        return

    data_list = get_scrambled_datasets(
        seasons,
        n_signal=n_injected,
        sources_in_hemisphere=hemisphere,
        number_of_sources=number_of_sources,
        dataset=dataset,
        indices=indices,
    )

    if len(indices) != len(data_list):
        raise Exception(
            f"Length of data list and length of indices should be the same. "
            f"len(datalist)={len(data_list)}; len(indices)={len(indices)}"
        )

    for i, dat in zip(indices, data_list):
        if use_skylab_splines:
            if isinstance(skylab_kernel, type(None)):
                raise ValueError(f"SkyLab kernel not given!")
            if (len(seasons) == 1) and (seasons[0] == "IC86_1"):
                llh_dict = get_llh_dict_for_skylab_splines(
                    unblind_llh,
                    skylab_kernel,
                    n_signal=ns,
                    number_of_sources=nsources,
                    hemisphere=hsphere,
                    gamma=gamma,
                    scramble_ind=i,
                )
            else:
                llh_dict = get_llh_dict_for_skylab_splines(unblind_llh, skylab_kernel)

        else:
            llh_dict = unblind_llh

        logger.debug(f"scanning likelihood for indice {i}")
        name = f"{directory}/{i}"
        mh_dict = get_minimizer_dict(
            name, number_of_sources, hemisphere, dat, llh_dict=llh_dict
        )
        ub = create_unblinder(mh_dict, mock_unblind=False, disable_warning=True)
        ub.scan_likelihood(scan_2d=True)


def make_summary_pdf(title, filename, diagnostic_plots_directory, llh_scans_directory):
    # make list of files to be included in the document
    files = list()
    captions = list()
    ts_file = f"{diagnostic_plots_directory}/TS.pdf"
    ts_caption = ""
    files.append(f"{diagnostic_plots_directory}/ns.pdf")
    captions.append("")
    files.append(f"{diagnostic_plots_directory}/gamma.pdf")
    captions.append("")
    files.append(f"{diagnostic_plots_directory}/TS_spatial.pdf")
    captions.append("TS when only using spatial info")
    files.append(f"{diagnostic_plots_directory}/ns_spatial.pdf")
    captions.append("ns when only using spatial info")

    indices = sorted(
        [
            int(sub_d)
            for sub_d in os.listdir(llh_scans_directory)
            if os.path.isdir(f"{llh_scans_directory}/{sub_d}")
        ]
    )

    for i in indices:
        files.append(f"{llh_scans_directory}/{i}/real_unblind/contour_scan.pdf")
        captions.append(f"Flaresatck LLH Scan of {i}")
        files.append(f"{llh_scans_directory}/{i}{skylab_llh_scan_str}.pdf")
        captions.append(f"SkyLab LLH Scan of {i}")

    # create document
    geometry_options = {"margin": "0.5in", "right": "2cm", "left": "2cm"}
    doc = pylatex.Document(
        filename, geometry_options=geometry_options, page_numbers=False
    )
    with doc.create(pylatex.Section(title, numbering=False)):
        with doc.create(pylatex.Figure(position="h!")) as fig:
            if os.path.isfile(ts_file):
                fig.add_image(ts_file, width=pylatex.NoEscape(r"0.49\linewidth"))
                fig.add_caption(ts_caption)

        for j in range(int(len(files) / 2)):
            with doc.create(pylatex.Figure(position="h!")):
                logger.debug("adding figure")

                with doc.create(
                    pylatex.SubFigure(
                        position="b", width=pylatex.NoEscape(r"0.49\linewidth")
                    )
                ) as left_fig:
                    if os.path.isfile(files[j * 2]):
                        left_fig.add_image(
                            files[j * 2]
                        )  # , width=pylatex.NoEscape(r'\linewidth'))
                        left_fig.add_caption(captions[j * 2])
                with doc.create(
                    pylatex.SubFigure(
                        position="b", width=pylatex.NoEscape(r"0.49\linewidth")
                    )
                ) as right_fig:
                    if os.path.isfile(files[j * 2 + 1]):
                        right_fig.add_image(
                            files[j * 2 + 1]
                        )  # , width=pylatex.NoEscape(r'\linewdith'))
                        right_fig.add_caption(captions[j * 2 + 1])

            doc.append(pylatex.VerticalSpace("1cm"))

    doc.generate_pdf(filename, clean_tex=True, clean=True, silent=True)


# ========================================       execute       ============================================ #


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)

    for d in [
        data_dir,
        this_cat_dir,
        plot_dir,
        zip_temp_dir,
    ]:  # , mh_plot_dir, this_storage_dir]:
        if not os.path.isdir(d):
            logger.debug(f"making directory {d}")
            os.makedirs(d)

    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--force_new", type=str, nargs="+", default=[""])
    parser.add_argument("--dont_ask", type=bool, nargs="?", default=False, const=True)
    args = parser.parse_args()

    # -----------------------------------------   use only one source   ----------------------------------- #

    outlier_indices_nsources = dict()

    for nsources in [5]:
        # plot sources
        for hsphere in hemispheres:
            cat = np.load(get_sources(hsphere, nsources))
            d, r = cat["dec_rad"], cat["ra_rad"] - np.pi
            plot_skymap(
                d, r, fn=f"{plot_dir}/{hsphere}_sources.pdf", title=f"{hsphere} sources"
            )

        use_dataset = ps_v002_p01.make_copy()

        # do analysis once with the Flaresatck smoothing method and once with the one employed in SkyLab
        smoothing_res_dict = dict()

        if nsources not in outlier_indices_nsources:
            outlier_indices_nsources[nsources] = dict()
        outlier_indices_dict = outlier_indices_nsources[nsources]

        for smoothing in [
            "skylab",
            "flarestack",
        ]:  # , 'skylab_splines_kernel0', 'skylab_splines_kernel1']:
            logger.info(f"using {smoothing} smoothing method")

            smoothing_str = (
                ""
                if (smoothing == "default") or (smoothing == "flarestack")
                else "SkyLab_pdf_smoothing"
                if smoothing == "skylab"
                else "SkyLab_splines_kernel1"
                if ("splines" in smoothing) and ("kernel1" in smoothing)
                else "SkyLab_splines_kernel0"
                if ("splines" in smoothing) and ("kernel0" in smoothing)
                else None
            )

            if isinstance(smoothing_str, type(None)):
                raise Exception(f"smoothing {smoothing} not understood!")

            if "splines" in smoothing:
                if "1" in smoothing:
                    skylab_kernel = 1
                elif "0" in smoothing:
                    skylab_kernel = 0
                else:
                    raise Exception(
                        f"Smoothing {smoothing} specifies skylab splines without kernel"
                    )
            else:
                skylab_kernel = None

            this_storage_dir = f"{storage_dir}{raw}{smoothing_str}/{minimizer_name}"
            raw_mh = f"{raw}{smoothing_str}/{minimizer_name}"
            mh_plot_dir = plot_output_dir(raw_mh)

            if not os.path.isdir(this_storage_dir):
                os.makedirs(this_storage_dir)

            # ---------- analyse individual seasons with only one source ---------- #

            individual_seasons_result_filename = (
                f"{this_storage_dir}/individual_seasons_{nsources}sources.json"
            )
            if (not os.path.isfile(individual_seasons_result_filename)) or (
                "fits" in args.force_new
            ):
                individual_seasons_res = dict()

                for season in use_dataset.keys():
                    # if smoothing == 'skylab_splines':
                    #     logger.info('copying SkyLab splines to Flaresatck Scratch Dir')
                    #     pathh = SoB_spline_path(use_dataset.get_single_season(season))
                    #
                    #     if not os.path.isdir(os.path.dirname(pathh)):
                    #         os.makedirs(os.path.dirname(pathh))
                    #
                    #     with open(pathh, 'wb') as f:
                    #         pickle.dump(get_skylab_splines(season, skylab_kernel), f)

                    hsphere_dict = dict()

                    for hsphere in ["northern", "southern"]:
                        catalogue_path = get_sources(hsphere, nsources)
                        catalogue = pd.DataFrame(np.load(catalogue_path))
                        logger.debug(f"\n{catalogue}")

                        ns_dict = dict()

                        for ns in [0, 5]:  # n_signal_range:
                            data_list = get_scrambled_datasets(
                                [season],
                                n_signal=ns,
                                sources_in_hemisphere=hsphere,
                                number_of_sources=nsources,
                                dataset=use_dataset,
                                check="y",
                                dont_check_backup_copy=args.dont_ask,
                            )

                            pdf_values_res = get_pdf_values(
                                scramble_indice,
                                season,
                                ns,
                                hsphere,
                                nsources,
                                gamma,
                                use_skylab_energy_splines=True
                                if "skylab_splines" in smoothing
                                else False,
                                skylab_kernel=skylab_kernel,
                            )

                            ns_dict[ns] = {
                                "TS": list(),
                                "ns": list(),
                                "gamma": list(),
                                "TS_spatial": list(),
                                "ns_spatial": list(),
                                f"pdf_values_scramble{scramble_indice}": pdf_values_res,
                            }

                            for ii, data in enumerate(data_list):
                                if "splines" in smoothing:
                                    if season == "IC86_1_":
                                        logger.debug("using skylab splines")
                                        llh_dict = get_llh_dict_for_skylab_splines(
                                            unblind_llh,
                                            skylab_kernel,
                                            n_signal=ns,
                                            number_of_sources=nsources,
                                            hemisphere=hsphere,
                                            gamma=gamma,
                                            scramble_ind=ii,
                                        )
                                        logger.debug("here here")
                                    else:
                                        logger.debug("here")
                                        llh_dict = get_llh_dict_for_skylab_splines(
                                            unblind_llh, skylab_kernel
                                        )

                                    llh_dict["smoothing_order"] = "skylab"
                                    llh_dict["gamma_precision"] = "flarestack"

                                else:
                                    llh_dict = copy.deepcopy(unblind_llh)
                                    llh_dict["smoothing_order"] = smoothing
                                    llh_dict["gamma_precision"] = smoothing

                                name = f"{raw}{smoothing_str}/{minimizer_name}/{season}/{hsphere}/{ns}/{ii}"

                                # unblind dictionary for minimizing with energy pdf
                                unblind_dict = get_minimizer_dict(
                                    name, nsources, hsphere, data, llh_dict
                                )

                                # this is ok because we are using scrambled data anyway!
                                ub = create_unblinder(
                                    unblind_dict,
                                    mock_unblind=False,
                                    disable_warning=True,
                                )

                                # unblind dictionary for minimizing only with spatial info
                                unblind_dict_spatial = get_minimizer_dict(
                                    name + "_spatial",
                                    nsources,
                                    hsphere,
                                    data,
                                    unblind_llh_spatial,
                                )

                                ub_spatial = create_unblinder(
                                    unblind_dict_spatial,
                                    mock_unblind=False,
                                    disable_warning=True,
                                )

                                ns_dict[ns]["TS"].append(ub.res_dict["TS"])
                                ns_dict[ns]["ns"].append(
                                    ub.res_dict["Parameters"]["n_s"]
                                )
                                ns_dict[ns]["gamma"].append(
                                    ub.res_dict["Parameters"]["gamma"]
                                )
                                ns_dict[ns]["TS_spatial"].append(
                                    ub_spatial.res_dict["TS"]
                                )
                                ns_dict[ns]["ns_spatial"].append(
                                    ub_spatial.res_dict["Parameters"]["n_s"]
                                )

                        hsphere_dict[hsphere] = ns_dict

                    individual_seasons_res[season] = hsphere_dict

                logger.debug(
                    f"saving results for individual seasons to {individual_seasons_result_filename}"
                )
                with open(individual_seasons_result_filename, "w") as f:
                    json.dump(individual_seasons_res, f)

            else:
                logger.debug(f"loading {individual_seasons_result_filename}")
                with open(individual_seasons_result_filename, "r") as f:
                    individual_seasons_res = json.load(f)

            smoothing_res_dict[smoothing] = dict()
            smoothing_res_dict[smoothing]["individual"] = individual_seasons_res

            # ---------- analyse multiple seasons  ---------- #

            multiple_seasons_result_filename = (
                f"{this_storage_dir}/multiple_seasons_{nsources}sources.json"
            )
            if (not os.path.isfile(multiple_seasons_result_filename)) or (
                "fits" in args.force_new
            ):
                multiple_seasons_hspere_dict = dict()
                for hsphere in ["southern", "northern"]:
                    catalogue_path = get_sources(hsphere, nsources)

                    multiple_seasons_ns_dict = dict()
                    for ns in [0, 5]:
                        # use_dataset.keys() will give a list of all seasons in the dataset
                        data_list = get_scrambled_datasets(
                            list(use_dataset.keys()),
                            n_signal=ns,
                            sources_in_hemisphere=hsphere,
                            number_of_sources=nsources,
                            dataset=use_dataset,
                            check="y",
                        )

                        multiple_seasons_ns_dict[ns] = {
                            "TS": list(),
                            "ns": list(),
                            "gamma": list(),
                        }

                        for ii, data in enumerate(data_list):
                            llh_dict = (
                                unblind_llh
                                if not "splines" in smoothing
                                else get_llh_dict_for_skylab_splines(
                                    unblind_llh, skylab_kernel
                                )
                            )

                            name = f"{raw}{smoothing_str}/{minimizer_name}/all_seasons/{hsphere}/{ns}/{ii}"

                            unblind_dict = {
                                "name": name,
                                "mh_name": minimizer_name,
                                "dataset": data,
                                "catalogue": catalogue_path,
                                "llh_dict": llh_dict,
                            }

                            # this is ok because we are using scrambled data anyway!
                            ub = create_unblinder(
                                unblind_dict, mock_unblind=False, disable_warning=True
                            )

                            multiple_seasons_ns_dict[ns]["TS"].append(ub.res_dict["TS"])
                            multiple_seasons_ns_dict[ns]["ns"].append(
                                ub.res_dict["Parameters"]["n_s"]
                            )
                            multiple_seasons_ns_dict[ns]["gamma"].append(
                                ub.res_dict["Parameters"]["gamma"]
                            )

                    multiple_seasons_hspere_dict[hsphere] = multiple_seasons_ns_dict

                logger.debug(
                    f"saving results for multiple seasons under {multiple_seasons_result_filename}"
                )
                with open(multiple_seasons_result_filename, "w") as f:
                    json.dump(multiple_seasons_hspere_dict, f)

            else:
                logger.debug(f"loading {multiple_seasons_result_filename}")
                with open(multiple_seasons_result_filename, "r") as f:
                    multiple_seasons_hspere_dict = json.load(f)

            smoothing_res_dict[smoothing]["multiple"] = multiple_seasons_hspere_dict

            # -----------------------     get skylab results     --------------------------- #

            for skylab_kernel in [0, 1]:
                if skylab_kernel not in outlier_indices_dict:
                    outlier_indices_dict[skylab_kernel] = dict()
                outlier_indices_skylab_kernel = outlier_indices_dict[skylab_kernel]

                skylab_individual_seasons_res = get_skylab_results(
                    f"individual_seasons_{nsources}sources_kernel{skylab_kernel}.json"
                )

                skylab_multiple_seasons_res = get_skylab_results(
                    f"multiple_seasons_{nsources}sources_kernel{skylab_kernel}.json"
                )

                skylab_individual_seasons_spatial_res = get_skylab_results(
                    f"individual_seasons_{nsources}sources_spatial_dang5.json"
                )

                skylab_multiple_seasons_spatial_res = get_skylab_results(
                    f"multiple_seasons_{nsources}sources_spatial_dang5.json"
                )

                skylab_pdf_values = get_skylab_results(
                    f"individual_seasons_{nsources}_pdf_values_scramble{scramble_indice}_kernel{skylab_kernel}.zip"
                )

                logger.debug(skylab_pdf_values.keys())

                # ---------------------     make plots    ------------------------ #

                # plot results for individual seasons
                for season, hsphere_dict in individual_seasons_res.items():
                    if season not in outlier_indices_skylab_kernel:
                        outlier_indices_skylab_kernel[season] = dict()
                    outlier_indices_season = outlier_indices_skylab_kernel[season]

                    logger.debug(f"plotting results for season {season}")
                    skylab_hsphere_dict = skylab_individual_seasons_res.get(
                        season, dict()
                    )
                    skylab_spatial_hsphere_dict = (
                        skylab_individual_seasons_spatial_res.get(season, dict())
                    )
                    skylab_pdf_vals_hsphere_dict = skylab_pdf_values.get(season, dict())
                    pdir = f"skylab_kernel{skylab_kernel}/{nsources}sources/{season}"

                    for hsphere, ns_dict in hsphere_dict.items():
                        if hsphere not in outlier_indices_season:
                            outlier_indices_season[hsphere] = dict()
                        outlier_indices_hsphere = outlier_indices_season[hsphere]

                        logger.debug(f"plotting results for {hsphere} hemisphere")
                        skylab_ns_dict = skylab_hsphere_dict.get(hsphere, dict())
                        skylab_spatial_ns_dict = skylab_spatial_hsphere_dict.get(
                            hsphere, dict()
                        )
                        skylab_pdf_vals_ns_dict = skylab_pdf_vals_hsphere_dict.get(
                            hsphere, dict()
                        )
                        plot_hsphere_dir = f"{pdir}/{hsphere}hemisphere"

                        for ns, params_with_pdf_values in ns_dict.items():
                            params = {
                                k: v
                                for k, v in params_with_pdf_values.items()
                                if "pdf" not in k
                            }
                            skylab_params = skylab_ns_dict.get(f"{ns}", dict())
                            skylab_spatial_params = skylab_spatial_ns_dict.get(
                                f"{ns}", dict()
                            )
                            skylab_pdf_values_dict = skylab_pdf_vals_ns_dict.get(
                                f"{ns}", dict()
                            )
                            logger.debug(f"plotting results for ns={ns}")

                            logger.debug(
                                f"SkyLab spatial params: {skylab_spatial_params.keys()}"
                            )
                            for k, l in skylab_spatial_params.items():
                                skylab_params[k + "_spatial"] = l

                            base = f"{plot_hsphere_dir}/ns{ns}"
                            plot_ns_dir = f"{mh_plot_dir}/{base}"
                            if not os.path.isdir(plot_ns_dir):
                                os.makedirs(plot_ns_dir)

                            # plot pdf values skylab against flarestack
                            flarestack_pdf_values_dict = params_with_pdf_values[
                                f"pdf_values_scramble{scramble_indice}"
                            ]
                            filename = (
                                f"{plot_ns_dir}/pdf_values_scramble{scramble_indice}"
                            )

                            make_pdf_values_plots(
                                flarestack_pdf_values_dict,
                                skylab_pdf_values_dict,
                                filename,
                                scramble_indice,
                                filename=f"{plot_ns_dir}/"
                                f"pdf_values_{nsources}sources_{season}_{hsphere}_{ns}_scramble{scramble_indice}",
                                nsources=nsources,
                                hemisphere=hsphere,
                                nsignal=ns,
                                season=season,
                                use_dataset=use_dataset,
                            )

                            # plot resulting parameters for each scramble
                            filename = f"{plot_ns_dir}/individual_scrambled_ns{ns}_{hsphere}hemisphere_{season}.pdf"
                            make_scramble_line_plot(
                                params,
                                skylab_params,
                                filename,
                                float(ns) * nsources,
                                gamma,
                            )

                            # plot resulting parameters in scatter plot
                            filename_base = f"{plot_ns_dir}/individual_scrambled_scatter_ns{ns}_{hsphere}hemisphere_{season}"
                            make_combined_scatter_plot(
                                params, skylab_params, f"{filename_base}.pdf"
                            )
                            # plot parameters in seperate scatter plot with histograms at the sides
                            make_individual_scatter_plots(
                                params,
                                skylab_params,
                                filename_base,
                                float(ns) * nsources,
                                gamma,
                                add_to_title=f"{season}\n",
                            )

                            raw_llh_scan_dir = f"{raw_mh}/{base}/llh_scans"
                            outlier_indices = get_outliers(params, skylab_params)

                            if not isinstance(outlier_indices, type(None)):
                                if ns not in outlier_indices_hsphere:
                                    outlier_indices_hsphere[ns] = list()
                                outlier_indices_ns = outlier_indices_hsphere[ns]
                                outlier_indices_ns.extend(
                                    [
                                        i
                                        for i in outlier_indices
                                        if i not in outlier_indices_ns
                                    ]
                                )

                            if (not isinstance(outlier_indices, type(None))) and (
                                scramble_indice not in outlier_indices
                            ):
                                outlier_indices.append(scramble_indice)
                            # directory, seasons, n_injected, hemisphere, number_of_sources, dataset, indices
                            make_llh_scans_of_outliers(
                                raw_llh_scan_dir,
                                [season],
                                ns,
                                hsphere,
                                nsources,
                                use_dataset,
                                outlier_indices,
                                use_skylab_splines=True
                                if "splines" in smoothing_str
                                else False,
                                skylab_kernel=skylab_kernel,
                            )

                            llh_scan_dir = plot_output_dir(raw_llh_scan_dir)
                            if not os.path.isdir(llh_scan_dir):
                                os.makedirs(llh_scan_dir)
                            copy_skylab_llh_scans(
                                outlier_indices,
                                llh_scan_dir,
                                ns=ns,
                                season_name=season,
                                hemisphere=hsphere,
                                nsources=nsources,
                                kernel=skylab_kernel,
                            )

                            make_summary_pdf(
                                title=f"{season}, ns={ns}, {hsphere}hemisphere, {nsources} sources",
                                filename=f"{plot_ns_dir}/{nsources}_{season}_{hsphere}_{ns}",
                                diagnostic_plots_directory=filename_base,
                                llh_scans_directory=plot_output_dir(raw_llh_scan_dir),
                            )

                            if smoothing == "flarestack":
                                logger.debug(
                                    smoothing_res_dict["skylab"]["individual"][season][
                                        hsphere
                                    ].keys()
                                )
                                make_plots_comparing_smoothings(
                                    f"{plot_ns_dir}/compare_smoothings",
                                    params,
                                    smoothing_res_dict["skylab"]["individual"][season][
                                        hsphere
                                    ][ns],
                                    skylab_params,
                                    ns,
                                    gamma,
                                )

                # plot results for multiple seasons
                for hsphere, ns_dict in multiple_seasons_hspere_dict.items():
                    logger.debug(
                        f"plotting results for multiple seasons and {hsphere} hemisphere"
                    )
                    skylab_ns_dict = skylab_multiple_seasons_res.get(hsphere, dict())
                    plot_hsphere_dir = f"skylab_kernel{skylab_kernel}/{nsources}sources/all_seasons/{hsphere}hemisphere"

                    for ns, params in ns_dict.items():
                        skylab_params = skylab_ns_dict.get(f"{ns}", dict())
                        logger.debug(f"plotting results for ns={ns}")

                        base = f"{plot_hsphere_dir}/ns{ns}"
                        plot_ns_dir = f"{mh_plot_dir}/{base}"
                        if not os.path.isdir(plot_ns_dir):
                            os.makedirs(plot_ns_dir)

                        # plot resulting parameters for each scramble
                        filename = f"{plot_ns_dir}/individual_scrambled_ns{ns}_{hsphere}hemisphere_all_seasons.pdf"
                        make_scramble_line_plot(
                            params,
                            skylab_params,
                            filename,
                            float(ns) * len(use_dataset.keys()),
                            gamma,
                        )

                        # plot resulting parameters in scatter plot
                        filename_base = f"{plot_ns_dir}/individual_scrambled_scatter_ns{ns}_{hsphere}hemisphere_all_seasons"
                        make_combined_scatter_plot(
                            params, skylab_params, f"{filename_base}.pdf"
                        )
                        # plot parameters in seperate scatter plot with histograms at the sides
                        make_individual_scatter_plots(
                            params,
                            skylab_params,
                            filename_base,
                            float(ns) * len(use_dataset.keys()),
                            gamma,
                            add_to_title=f"all seasons\n",
                        )

                        raw_llh_scan_dir = f"{raw_mh}/{base}/llh_scans"
                        outlier_indices = get_outliers(params, skylab_params)
                        make_llh_scans_of_outliers(
                            raw_llh_scan_dir,
                            list(use_dataset.keys()),
                            ns,
                            hsphere,
                            nsources,
                            use_dataset,
                            outlier_indices,
                            use_skylab_splines=True
                            if "splines" in smoothing_str
                            else False,
                            skylab_kernel=skylab_kernel,
                        )

                        llh_scan_dir = plot_output_dir(raw_llh_scan_dir)
                        if not os.path.isdir(llh_scan_dir):
                            os.makedirs(llh_scan_dir)
                        copy_skylab_llh_scans(
                            outlier_indices,
                            plot_output_dir(raw_llh_scan_dir),
                            ns=ns,
                            season_name="all_seasons",
                            hemisphere=hsphere,
                            nsources=nsources,
                            kernel=skylab_kernel,
                        )

                        make_summary_pdf(
                            title=f"all seasons, ns={ns}, {hsphere}hemisphere, {nsources} sources",
                            filename=f"{plot_ns_dir}/{nsources}_all_seasons_{hsphere}_{ns}",
                            diagnostic_plots_directory=filename_base,
                            llh_scans_directory=plot_output_dir(raw_llh_scan_dir),
                        )

                        if smoothing == "flarestack":
                            try:
                                make_plots_comparing_smoothings(
                                    f"{plot_ns_dir}/compare_smoothings",
                                    params,
                                    smoothing_res_dict["skylab"]["multiple"][hsphere][
                                        ns
                                    ],
                                    skylab_params,
                                    float(ns) * len(use_dataset.keys()),
                                    gamma,
                                )
                            except KeyError as e:
                                logger.error(e)

    save_outlier_dict(outlier_indices_nsources)

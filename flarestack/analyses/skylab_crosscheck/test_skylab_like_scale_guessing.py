import numpy as np
import argparse
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01, diffuse_8_year
from flarestack.shared import plot_output_dir, flux_to_k, catalogue_dir
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
from flarestack.analyses.skylab_crosscheck.skylab_crosscheck_with_same_scramble import (
    make_sources,
)
from flarestack.analyses.skylab_crosscheck.data import (
    get_skylab_crosscheck_stacking_same_sources_res,
)
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.utils.prepare_catalogue import custom_sources
import os
import logging


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--analyse", type=bool, default=False, nargs="?", const=True)
parser.add_argument("-s", "--seasons", type=str, default=["IC86_1"], nargs="+")
parser.add_argument("--smoothing", type=str, default="flarestack")
args = parser.parse_args()

logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")
logger = logging.getLogger("main")

# use_precision(args.smoothing)
# use_smoothing(args.smoothing)


injection_energy = {"energy_pdf_name": "power_law"}

injection_time = {"time_pdf_name": "steady"}

llh_energy = {"energy_pdf_name": "power_law"}

llh_time = {"time_pdf_name": "steady"}

cluster = 500
ntrials = 500

if cluster and (cluster > ntrials):
    raise ValueError(
        f"Number of trials {ntrials} is smaller than number of cluster jobs {cluster}!"
    )

# Spectral indices to loop over
gammas = [2.0]

# set common ref time mjd
ref_time_mjd = np.nan

# Base name
mh_name = "fixed_weights"
weighted = False
ps = False
seasons = args.seasons

raw = f"analyses/skylab_crosscheck/crosscheck_stacking_same_test_scale_guessing/{mh_name}_{args.smoothing}-smoothing"

if weighted:
    raw += "_weighted"

if ps:
    raw += "_ps"

if seasons:
    for s in seasons:
        raw += f"_{s}"

if not os.path.exists(plot_output_dir(raw)):
    os.makedirs(plot_output_dir(raw))

full_res = dict()

job_ids = []

# Loop over SN catalogues
if __name__ == "__main__":
    res = dict()

    for hsphere in ["northern", "southern"]:
        hsphere_name = f"{raw}/{hsphere}"

        # this_sindecs, this_ras = np.array(sindecs[hsphere])[:5], np.array(ras[hsphere])[:5]
        # names = [str(sd) for sd in this_sindecs]
        # weights = distances = [1] * len(this_sindecs)
        # catalogue = custom_sources(names, this_ras, this_sindecs, weights, distances)
        catalogue = make_sources(hsphere, 5)
        cat_path = f"{catalogue_dir}{raw}/{hsphere}_cat.npy"

        cat_dir = dir_path = os.path.dirname(os.path.realpath(cat_path))
        if not os.path.isdir(cat_dir):
            logger.debug(f"Making directory {cat_dir}")
            os.makedirs(cat_dir)

        np.save(cat_path, catalogue)

        gamma_res = dict()

        for gamma in gammas:
            injection_energy["gamma"] = gamma

            inj_dict = {
                "injection_energy_pdf": injection_energy,
                "injection_sig_time_pdf": injection_time,
            }

            llh_dict = {
                "llh_name": "standard"
                if not mh_name == "large_catalogue"
                else "standard_matrix",
                "llh_energy_pdf": llh_energy,
                "llh_sig_time_pdf": llh_time,
                "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
                "gamma_precision": args.smoothing,
                "smoothing_order": args.smoothing,
            }

            full_name = f"{hsphere_name}/{gamma:.2f}"

            length = 365 * 7

            scale = (
                flux_to_k(
                    reference_sensitivity(np.sin(0.5), gamma=gamma)
                    * 40
                    * math.sqrt(float(len(catalogue)))
                )
                * 200.0
            ) / length

            if hsphere == "southern":
                scale *= 5

                if seasons == ["IC40"]:
                    scale *= 4

            data = ps_v002_p01
            if seasons:
                data = data.get_seasons(*seasons)

            dataset = custom_dataset(data, catalogue, llh_dict["llh_sig_time_pdf"])
            logger.debug(f"{dataset.keys()}")

            mh_dict = {
                "name": full_name,
                "mh_name": mh_name,
                "dataset": dataset,
                "catalogue": cat_path,
                "inj_dict": inj_dict,
                "llh_dict": llh_dict,
                "fixed_scale": 0,
                "n_trials": ntrials / cluster if cluster else ntrials,
                "n_steps": 10,
                "allow_extrapolated_sensitivity": False,
            }

            job_id = None
            if args.analyse:
                job_id = analyse(
                    mh_dict,
                    cluster=True if cluster else False,
                    n_cpu=1 if cluster else 16,
                    n_jobs=cluster,
                    h_cpu="04:59:59",
                )
            job_ids.append(job_id)

            gamma_res[gamma] = mh_dict

        res[hsphere] = gamma_res

    if cluster and np.any(job_ids):
        logger.info(f"waiting for jobs {job_ids}")
        wait_for_cluster(job_ids)

    stacked_sens_flux = {}

    # calculating sensitivities
    for hsphere, gamma_res in res.items():
        stacked_sens_flux[hsphere] = dict()
        for gamma, rh_dict in gamma_res.items():
            rh = ResultsHandler(rh_dict)

            stacked_sens_flux[hsphere][gamma] = {
                "sens": rh.sensitivity,
                "sens_e": rh.sensitivity_err,
            }

    # making plots
    sl_res = get_skylab_crosscheck_stacking_same_sources_res(
        "manualmedianPointSourceTracks_v002p01", seasons
    )
    gamma_color = {2.0: "blue"}
    for hsphere, gamma_res in stacked_sens_flux.items():
        fig, (ax, ax2) = plt.subplots(
            2, gridspec_kw={"height_ratios": [3, 1]}, sharex="all"
        )

        for gamma, sens_dict in gamma_res.items():
            yerr = np.atleast_2d(sens_dict["sens_e"]).T
            logger.debug(f"error shape is {np.shape(yerr)}")
            ax.errorbar(
                ["flarestack"], sens_dict["sens"], yerr=yerr, marker="o", color="blue"
            )
            y1_fs = sens_dict["sens"] - sens_dict["sens_e"][0]
            y2_fs = sens_dict["sens"] + sens_dict["sens_e"][1]
            ax.fill_between(
                ["flarestack", "skylab"], y1=y1_fs, y2=y2_fs, color="blue", alpha=0.2
            )

            for kernel in [1]:
                if sl_res:
                    this_sl_res = sl_res[kernel][hsphere][gamma]
                    ax.errorbar(
                        ["skylab"],
                        this_sl_res["sens"],
                        yerr=this_sl_res["sens_e"],
                        marker="o",
                        color="red",
                        alpha=0.5 + kernel / 2,
                    )
                    y1_sl = this_sl_res["sens"] - this_sl_res["sens_e"]
                    y2_sl = this_sl_res["sens"] + this_sl_res["sens_e"]
                    ax.fill_between(
                        ["flarestack", "skylab"],
                        y1=y1_sl,
                        y2=y2_sl,
                        color="red",
                        alpha=0.2,
                    )

                    ax2.errorbar(
                        ["flarestack"],
                        sens_dict["sens"] / this_sl_res["sens"],
                        yerr=yerr / this_sl_res["sens"],
                        marker="o",
                        color="blue",
                    )
                    ax2.fill_between(
                        ["flarestack", "skylab"],
                        y1=(sens_dict["sens"] - yerr[0]) / this_sl_res["sens"],
                        y2=(sens_dict["sens"] + yerr[1]) / this_sl_res["sens"],
                        color="blue",
                        alpha=0.2,
                    )

                    ax2.errorbar(
                        ["skylab"],
                        this_sl_res["sens"] / this_sl_res["sens"],
                        yerr=this_sl_res["sens_e"] / this_sl_res["sens"],
                        marker="o",
                        color="red",
                    )
                    ax2.fill_between(
                        ["flarestack", "skylab"],
                        y1=1 - this_sl_res["sens_e"] / this_sl_res["sens"],
                        y2=1 + this_sl_res["sens_e"] / this_sl_res["sens"],
                        color="red",
                        alpha=0.2,
                    )

                    # for y, c, x in zip([[y1_fs, sens_dict['sens'], y2_fs], [y1_sl, this_sl_res['sens'], y2_sl]],
                    #                 ['blue', 'red'],
                    #                 ['flarestack', 'skylab']):
                    #     y_rel = np.array([y[0]-y[1], y[1], y[2]+y[1]])/this_sl_res['sens']
                    #     yerr = np.atleast_2d([y_rel[0]+y_rel[1], y_rel[1]-y_rel[1]]).T
                    #     logger.debug(f'shape of yerr is {np.shape(yerr)}')
                    #     ax2.errorbar([x], y_rel[1], yerr=yerr, marker='o', color=c)
                    #     ax2.fill_between(['flarestack', 'skylab'], y1=y_rel[0]+y_rel[1], y2=y_rel[2]-y_rel[1], color=c, alpha=0.2)

                else:
                    logger.warning(
                        f"No SkyLab results for gamma={gamma:.2f} and kernel {kernel}"
                        f" in {hsphere} hemisphere"
                    )

            ax.set_ylabel("Sensitivity Flux [GeV sr$^{-1}}$ s$^{-1}$ cm$^{-2}$]")
            ax2.set_ylabel("ratio")
            ax2.set_yscale("log")
            ax2.axhline(1, ls="--", color="red")
            xlim = ax2.get_xlim()
            ax2.set_xlim((min(xlim) - 0.5), max(xlim) + 0.5)
            ax.set_title(f"$\gamma$={gamma:.2f}")

            plt.tight_layout()
            fn = f"{plot_output_dir(raw)}/{hsphere}_sensitivities.pdf"
            fig.savefig(fn)
            logger.debug(f"saving under {fn}")
            plt.close()

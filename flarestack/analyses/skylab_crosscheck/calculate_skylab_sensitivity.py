"""Script to calculate the sensitivity and discovery potential for some sources to check consistency with skylab.
"""
import numpy as np
import argparse
from flarestack.analyses.skylab_crosscheck.make_sources import fs_sources, nsources
from flarestack.analyses.skylab_crosscheck.skylab_results import sl_data_dir
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import analyse, wait_for_cluster
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--analyse", type=bool, default=False, const=True, nargs="?")
args = parser.parse_args()


logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")

# set up result dictionaries
full_res = dict()
analyses = dict()

# specifiy the minimizer
mh_name = "large_catalogue"

# sub path in plot output directory
raw = "analyses/skylab_crosscheck/" + mh_name + "/"
# data_dir = os.environ['HOME'] + '/flarestack_cc/'


# define a function to make a
def plot_ra_distribution(sin_dec, n_sources):
    filename = plot_output_dir(
        f"{raw}{sin_dec:.4f}/{int(n_sources)}sources/ra_distribution.pdf"
    )
    title = rf"sin($\delta$)={sin_dec:.2f} \n n={n_sources:.0f}"

    catalogue_filename = fs_sources(str(int(n_sources)), sin_dec)
    c = np.load(catalogue_filename)

    ra = c["ra_rad"]
    f, a = plt.subplots()
    a.hist(ra)
    a.set_xlabel("right ascensions")
    a.set_title(title)
    f.savefig(filename)
    plt.close()


same_sindecs = [-0.5, -0.25, 0, 0.25, 0.5, 0.75]
# same_sindecs = [0.25]
cluster = 100
ntrials = 100

# Initialise Injectors/LLHs

gammas = [2.0]

res = dict()
job_ids = list()

for smoothing in ["flarestack", "skylab"]:
    smoothing_name = f"{raw}{smoothing}-smoothing/"
    gamma_res = dict()

    for gamma in gammas:
        gamma_name = f"{smoothing_name}{gamma:.2f}/"

        logging.info(f"gamma = {gamma:.2f}")

        injection_energy = {
            "energy_pdf_name": "power_law",
            "gamma": gamma,
        }

        llh_time = {
            "time_pdf_name": "steady",
        }

        injection_time = {
            "time_pdf_name": "steady",
        }

        llh_dict = {
            "llh_name": "standard_matrix",
            "llh_energy_pdf": injection_energy,
            "llh_sig_time_pdf": llh_time,
            "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
            "smoothing_order": smoothing,
            "gamma_precision": smoothing,
        }

        inj_dict = {
            "injection_energy_pdf": injection_energy,
            "injection_sig_time_pdf": injection_time,
        }

        length = 0
        for season_name, season in ps_v002_p01.seasons.items():
            length += season.get_time_pdf().get_livetime()

        logging.info("injection length in livetime is {:.2f}".format(length))

        sin_res = dict()

        for sindec in same_sindecs:
            full_res = dict()

            for i, n in enumerate(np.array(nsources)):
                logging.info(f"stacking {n} sources")
                logging.info(f"cat path is {fs_sources(n, sindec)}")

                name = (
                    gamma_name + "{:.4f}/".format(sindec) + str(n) + "sources"
                    if sindec is not None
                    else gamma_name + "None/" + str(n) + "sources"
                )
                catalogue = np.load(fs_sources(n, sindec))
                closest_src = np.sort(catalogue, order="distance_mpc")[0]

                scale = (
                    flux_to_k(
                        reference_sensitivity(
                            np.sin(closest_src["dec_rad"]), gamma=gamma
                        )
                        * 40
                        * (math.log(float(len(catalogue)), 4) + 1)
                    )
                    * 200.0
                ) / length

                mh_dict = {
                    "name": name,
                    "mh_name": mh_name,
                    "dataset": custom_dataset(
                        ps_v002_p01, catalogue, llh_dict["llh_sig_time_pdf"]
                    ),
                    "catalogue": fs_sources(n, sindec),
                    "inj_dict": inj_dict,
                    "llh_dict": llh_dict,
                    "scale": scale,
                    "n_trials": ntrials / cluster if cluster else ntrials,
                    "n_steps": 10,
                }

                job_id = None
                if args.analyse:
                    job_id = analyse(
                        mh_dict,
                        cluster=True if cluster else False,
                        n_cpu=1 if cluster else 25,
                        n_jobs=cluster,
                        h_cpu="00:59:59",
                    )
                job_ids.append(job_id)

                full_res[str(n)] = mh_dict

            sin_res[str(sindec)] = full_res

        gamma_res[gamma] = sin_res

    res[smoothing] = gamma_res

if cluster and np.any(job_ids):
    logging.info(f"waiting for jobs {job_ids}")
    wait_for_cluster(job_ids)

for smoothing, gamma_res in res.items():
    for gamma, sin_res in gamma_res.items():
        for sindec in same_sindecs:
            full_res = sin_res[str(sindec)]

            sens = [[], [], []]

            for n in full_res:
                logging.debug(f"n = {n}, type={type(n)}")

                rh_dict = full_res[n]
                rh = ResultsHandler(rh_dict)

                sens[0].append(float(n))
                sens[1].append(rh.sensitivity)
                sens[2].append(rh.sensitivity_err)

                plot_ra_distribution(sindec, float(n))

            # load results from skylab if they exist
            skylab_result_path = sl_data_dir(
                sindec
            ) + "/nsources_gamma{:.1f}.npy".format(gamma)
            skylab_results = (
                np.load(skylab_result_path)
                if os.path.isfile(skylab_result_path)
                else None
            )

            # normalize all points to
            norm_to = reference_sensitivity(sindec, gamma=gamma)

            fig, ax = plt.subplots()

            Nsqrt = np.sqrt(np.array(sens[0]))
            Nflat = [1.0] * len(sens[0])

            ax.plot(sens[0], Nsqrt, "k--", label=r"$F \sim \sqrt{N}$")
            ax.plot(sens[0], Nflat, "k-.", label=r"$F = const$")
            ax.errorbar(
                sens[0],
                # np.array(sens[1]) / sens[1][0],
                np.array(sens[1]) / norm_to,
                yerr=np.array(sens[2]).T / norm_to,
                marker="",
                ls="-",
                capsize=3,
                label=f"flarestack",
            )

            if skylab_results is not None:
                logging.info("drawing skylab results")
                ax.errorbar(
                    skylab_results["nsources"],
                    # skylab_results['sensitivity'] / skylab_results['sensitivity'][0],
                    skylab_results["sensitivity"] / norm_to,
                    yerr=skylab_results["sensitivity_error"] / norm_to,
                    marker="",
                    ls="-",
                    capsize=3,
                    label="skylab",
                )
            else:
                logging.info("no skylab results")

            ax.set_xlabel("$N$")
            ax.set_ylabel(r"$F \, \cdot \, F_{\mathrm{point \, source}}^{-1}$")
            ax.set_xscale("log")
            ax.set_title(
                "stacked sensitivity \n" + r"$\sin(\delta)=${:.2f}".format(sindec)
            )
            ax.legend()

            plt.tight_layout()

            fig.savefig(
                plot_output_dir(raw + "/{:.4f}/".format(sindec))
                + f"sens_nsources_gamma={gamma}_"
                f"sindec{sindec:.2f}.pdf"
            )
            plt.close()

            fig2, ax2 = plt.subplots()

            Nsqrt = 1 / np.sqrt(sens[0])
            Nrez = 1 / np.array(sens[0])

            ax2.plot(sens[0], Nsqrt, "k--", label=r"$F \sim \sqrt{N}$")
            ax2.plot(sens[0], Nrez, "k-.", label=r"$F = const$")
            ax2.errorbar(
                sens[0],
                # np.array(sens[1]) / sens[1][0] / np.array(sens[0]),
                np.array(sens[1]) / norm_to / np.array(sens[0]),
                yerr=np.array(sens[2]).T / norm_to / np.array(sens[0]),
                marker="",
                ls="-",
                capsize=3,
                label="flarestack",
            )

            if skylab_results is not None:
                logging.info("drawing skylab results")
                ax2.errorbar(
                    skylab_results["nsources"],
                    # skylab_results['sensitivity'] / skylab_results['sensitivity'][0] / np.array(sens[0]),
                    skylab_results["sensitivity"] / norm_to / np.array(sens[0]),
                    yerr=skylab_results["sensitivity_error"]
                    / norm_to
                    / np.array(sens[0]),
                    marker="",
                    ls="-",
                    capsize=3,
                    label="skylab",
                )
            else:
                logging.info("no skylab results")

            ax2.set_xlabel("$N_{\mathrm{sources}}$")
            ax2.set_ylabel(
                r"$F \, \cdot \, F_{\mathrm{point \, source \, reference}}^{-1} \, \cdot \, N_{\mathrm{sources}}^{-1}$"
            )
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_title(
                "stacked sensitivity per source\n"
                + r"$\sin(\delta)=${:.2f}".format(sindec)
            )
            ax2.legend()

            plt.tight_layout()

            fig2.savefig(
                plot_output_dir(raw + "/{:.4f}/".format(sindec))
                + f"sens_nsources_gamma={gamma}_persource_"
                f"sindec{sindec:.2f}.pdf"
            )
            plt.close()

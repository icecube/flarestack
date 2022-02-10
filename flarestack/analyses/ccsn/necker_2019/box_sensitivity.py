"""Script to calculate the sensitivity and discovery potential for CCSNe.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01, ps_v002_p03
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import (
    limits,
    get_figure_limits,
    p_vals,
)
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import (
    sn_cats,
    updated_sn_catalogue_name,
    sn_time_pdfs,
    raw_output_dir,
    pdf_names,
    limit_sens,
)
from flarestack.analyses.ccsn import get_sn_color
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
import math
import pickle
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging
import time

# Set Logger Level

start = time.time()

logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")

# LLH Energy PDF
llh_energy = {
    "energy_pdf_name": "power_law",
}

cluster = 500

# Spectral indices to loop over
gammas = [2.0, 2.5]

# minimizer to use
mh_name = "fit_weights"

# using the box pdfs
pdf_type = "box"

# base name
raw = raw_output_dir + f"/calculate_sensitivity_ps-v002p03/{mh_name}/{pdf_type}/"

# set up emtpy dictionary to store the minimizer information in
full_res = dict()

# set up empty list for cluster job IDs
job_ids = []

plot_results = "_figureresults"

if __name__ == "__main__":

    # loop over SN catalogues
    for cat in sn_cats:

        name = raw + cat + "/"

        # set up empty results dictionary for this catalogue
        cat_res = dict()

        # get the time pdfs for this catalogue
        time_pdfs = sn_time_pdfs(cat, pdf_type=pdf_type)

        # Loop over time PDFs
        for llh_time in time_pdfs:

            # set up an empty results array for this time pdf
            time_res = dict()

            logging.debug(f"time_pdf is {llh_time}")

            time_key = str(llh_time["post_window"] + llh_time["pre_window"])

            pdf_time = (
                float(time_key) if llh_time["pre_window"] == 0 else -float(time_key)
            )
            pdf_name = pdf_names(pdf_type, pdf_time)
            cat_path = updated_sn_catalogue_name(cat)
            logging.debug("catalogue path: " + str(cat_path))

            # load catalogue and select the closest source
            # that serves for estimating a good injection scale later
            catalogue = np.load(cat_path)
            logging.debug("catalogue dtype: " + str(catalogue.dtype))
            closest_src = np.sort(catalogue, order="distance_mpc")[0]

            time_name = name + time_key + "/"

            # set up the likelihood dictionary
            llh_dict = {
                "llh_name": "standard",
                "llh_energy_pdf": llh_energy,
                "llh_sig_time_pdf": llh_time,
                "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
            }

            # set up an injection dictionary which will be equal to the time pdf dictionary
            injection_time = llh_time

            # Loop over spectral indices
            for gamma in gammas:

                full_name = time_name + str(gamma) + "/"

                length = float(time_key)

                # try to estimate a good scale based on the sensitivity from the 7-yr PS sensitivity
                # at the declination of the closest source
                scale = (
                    0.5
                    * (
                        flux_to_k(
                            reference_sensitivity(
                                np.sin(closest_src["dec_rad"]), gamma=gamma
                            )
                            * 40
                            * math.sqrt(float(len(catalogue)))
                        )
                        * 200.0
                    )
                    / length
                )

                # in some cases the sensitivity is outside the tested range
                # to get a good sensitivity, adjust the scale in these cases
                if cat == "IIP":
                    scale *= 0.2
                if cat == "Ibc":
                    scale *= 0.5
                if cat == "IIn" and gamma == 2.5 and pdf_time == 1000:
                    scale *= 1.4
                if cat == "IIP" and gamma == 2.5:
                    scale *= 0.6

                # set up an injection dictionary and set the desired spectral index
                injection_energy = dict(llh_energy)
                injection_energy["gamma"] = gamma

                inj_dict = {
                    "injection_energy_pdf": injection_energy,
                    "injection_sig_time_pdf": injection_time,
                    "poisson_smear_bool": True,
                }

                # set up the final minimizer dictionary
                mh_dict = {
                    "name": full_name,
                    "mh_name": mh_name,
                    "dataset": custom_dataset(
                        ps_v002_p03, catalogue, llh_dict["llh_sig_time_pdf"]
                    ),
                    "catalogue": cat_path,
                    "inj_dict": inj_dict,
                    "llh_dict": llh_dict,
                    "scale": scale,
                    "n_trials": 1000 / cluster if cluster else 500,
                    "n_steps": 10,
                }

                # call the main analyse function
                job_id = None
                job_id = analyse(
                    mh_dict,
                    cluster=True if cluster else False,
                    n_cpu=1 if cluster else 32,
                    n_jobs=cluster,
                    h_cpu="00:59:59",
                )
                job_ids.append(job_id)

                time_res[gamma] = mh_dict

            cat_res[time_key] = time_res

        full_res[cat] = cat_res

    # Wait for cluster. If there are no cluster jobs, this just runs
    if cluster and np.any(job_ids):
        logging.info(f"waiting for jobs {job_ids}")
        wait_for_cluster(job_ids)

    # set up an empty results dictionary
    stacked_sens_flux = {}

    # loop over the SN catalogues for getting the results
    for b, (cat_name, cat_res) in enumerate(full_res.items()):

        stacked_sens_flux[cat_name] = {}

        # loop over the analysed times
        for (time_key, time_res) in cat_res.items():

            stacked_sens_flux[cat_name][time_key] = {}

            # set up emtpy result lists
            sens_livetime = []
            fracs = []
            disc_pots_livetime = []
            sens_e = []
            disc_e = []

            labels = []

            # Loop over gamma
            for (gamma, rh_dict) in sorted(time_res.items()):

                logging.debug(
                    f"gamma is {gamma}, cat is {cat_name}, pdf time is {time_key}"
                )
                # calling ResultsHandler will calculate the overfluctuations and sensitivities
                rh = ResultsHandler(rh_dict)

                # get the injection time and convert it to seconds
                inj = rh_dict["inj_dict"]["injection_sig_time_pdf"]
                injection_length = inj["pre_window"] + inj["post_window"]
                inj_time = injection_length * 60 * 60 * 24

                # Convert IceCube numbers to astrophysical quantities
                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj_dict"]["injection_energy_pdf"]
                )

                key = "Energy Flux (GeV cm^{-2} s^{-1})"
                e_key = "Mean Luminosity (erg/s)"

                sens_livetime.append(astro_sens[key] * inj_time)
                disc_pots_livetime.append(astro_disc[key] * inj_time)

                sens_e.append(astro_sens[e_key] * inj_time)
                disc_e.append(astro_disc[e_key] * inj_time)

                fracs.append(gamma)

                stacked_sens_flux[cat_name][time_key][gamma] = (
                    rh.sensitivity,
                    rh.sensitivity_err,
                    astro_sens[e_key] * inj_time,
                )

            name = "{0}/{1}/{2}".format(raw, cat_name, time_key)

            # make plots total energy/fluence plots
            for j, [fluence, energy] in enumerate(
                [[sens_livetime, sens_e], [disc_pots_livetime, disc_e]]
            ):

                logging.debug("fluence: " + str(fluence))
                logging.debug("energy: " + str(energy))

                plt.figure()
                ax1 = plt.subplot(111)

                ax2 = ax1.twinx()

                # cols = ["#00A6EB", "#F79646", "g", "r"]
                linestyle = ["-", "-"][j]

                ax1.plot(
                    fracs,
                    fluence,
                    label=labels,
                    linestyle=linestyle,
                )
                ax2.plot(
                    fracs,
                    energy,
                    linestyle=linestyle,
                )

                y_label = [
                    r"Energy Flux (GeV cm^{-2} s^{-1})",
                    r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)",
                ]

                ax2.grid(True, which="both")
                ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
                ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
                ax1.set_xlabel(r"Spectral Index ($\gamma$)")
                ax1.set_yscale("log")
                ax2.set_yscale("log")

                for k, ax in enumerate([ax1, ax2]):
                    y = [fluence, energy][k]
                    logging.debug("y: " + str(y))

                    # ax.set_ylim(0.95 * min(y),
                    #             1.1 * max(y))

                plt.title(
                    "Stacked "
                    + ["Sensitivity", "Discovery Potential"][j]
                    + " for "
                    + cat_name
                    + " SNe"
                )

                plt.tight_layout()
                plt.savefig(
                    plot_output_dir(name)
                    + "/spectral_index_"
                    + ["sens", "disc"][j]
                    + "_"
                    + cat_name
                    + ".pdf"
                )
                plt.close()

    # ================================       save sensitivities        ============================== #

    with open(limit_sens(raw), "wb") as f:
        pickle.dump(stacked_sens_flux, f)

    # =================================        make final plots       =============================== #

    # loop over gammas to make a plots for each spectral indice
    for gamma in gammas:

        # -------------    plot sensitivity against box length   ------------------- #
        flux_fig, flux_ax = plt.subplots()

        # loop over SN catalogues and plot the sensitivities against the box length
        for cat_name, cat_dict in stacked_sens_flux.items():

            x_raw = [float(key) for key in cat_dict.keys()]
            x = np.array(x_raw)[np.argsort(x_raw)]

            y = [stacked_sens_flux[cat_name][key][gamma][0] for key in cat_dict.keys()]
            y = np.array(y)[np.argsort(x_raw)]

            yerr = [
                stacked_sens_flux[cat_name][key][gamma][1] for key in cat_dict.keys()
            ]
            yerr = np.array(yerr)[np.argsort(x_raw)]

            logging.debug("\n x: {0};      y: {1};       yerr: {2}".format(x, y, yerr))

            flux_ax.errorbar(
                x,
                y,
                yerr=np.array(yerr).T,
                ls="--",
                label=cat_name,
                marker="",
                color=get_sn_color(cat_name),
            )

        flux_ax.set_xscale("log")
        flux_ax.set_yscale("log")
        flux_ax.set_xlabel(r"$t_{\mathrm{decay}}$ in years")
        flux_ax.set_ylabel(r"flux in GeV$^{-1}$ s$^{-1}$ cm$^{-2}$")
        flux_ax.legend()

        fname_flux = plot_output_dir(
            raw + f"stacked_sensitivity_flux_gamma{gamma}_{mh_name}.pdf"
        )
        logging.debug(f"saving figure under {fname_flux}")
        flux_fig.savefig(fname_flux)

        stacked_sens = {
            cat_name: np.array(
                [
                    (stacked_sens_flux[cat_name][time_key][gamma][2], time_key)
                    for time_key in stacked_sens_flux[cat_name]
                ],
                dtype="<f8",
            )
            for cat_name in stacked_sens_flux
        }

        # -------------    plot total emitted energy against box length   ---------------- #
        fig, ax = plt.subplots()

        # loop over SN catalogues and plot the total emitted neutrino energy against the box length
        for cat_name in stacked_sens:
            plot_arr = np.array(stacked_sens[cat_name], dtype="<f8")
            logging.debug(f"plot array: \n {plot_arr}")

            yerr = plot_arr[:, 0] * 0.1
            patch = ax.errorbar(
                plot_arr[:, 1],
                plot_arr[:, 0],
                xerr=plot_arr[:, 1] * 0.05,
                yerr=yerr,
                uplims=True,
                ls="",
                markersize=1,
                capsize=0.5,
                color=get_sn_color(cat_name),
            )

            ax.plot(
                plot_arr[:, 1],
                plot_arr[:, 0] - yerr,
                marker="v",
                color=patch.lines[0].get_c(),
                label=cat_name,
                ls="",
            )

            pval_mask = (
                p_vals[pdf_type][cat_name if not "P" in cat_name else "IIp"]["pval"]
                >= 0.5
            )

            # plot the original results by Alex Stasik to make a comparison

            if plot_results == "_figureresults":
                res = get_figure_limits(cat_name, pdf_type)
                res_sens = res[pval_mask]
                res_no_sens = res[np.invert(pval_mask)]

                for arr, marker, label in zip(
                    [res_sens, res_no_sens], ["d", "x"], ["sensitivity", "limit"]
                ):
                    ax.errorbar(
                        arr["t"],
                        arr["E"],
                        xerr=arr["t"] * 0.05,
                        yerr=arr["E"] * 0.1,
                        uplims=True,
                        ls="",
                        markersize=1,
                        capsize=0.5,
                        color=patch.lines[0].get_c(),
                        alpha=0.5,
                    )
                    ax.plot(
                        arr["t"],
                        arr["E"] - arr["E"] * 0.1,
                        marker=marker,
                        color=patch.lines[0].get_c(),
                        label=cat_name + " Stasik " + label,
                        ls="",
                        alpha=0.5,
                    )

            if plot_results == "_tableresults":
                cat_key = cat_name if "p" not in cat_name else "IIP"
                ekey = "Fit" if "fit" or "Fit" in mh_name else "Fix"
                ekey += " Energy (erg)"
                ax.axhline(
                    limits[cat_key][ekey].value,
                    ls="--",
                    color=patch.lines[0].get_c(),
                    label=cat_name + " previous limit",
                )

        ax.set_ylabel("$E^{\\nu}_{tot}$ [erg]")
        ax.set_xlabel("Box function $\Delta T$ [d]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

        plt.grid()
        plt.title(f"Stacked sensitivity for $\gamma = {gamma}$")
        plt.tight_layout()

        plt.savefig(
            plot_output_dir(
                raw + f"stacked_sensitivity_gamma{gamma}_{mh_name}{plot_results}.pdf"
            )
        )

    end = time.time()

    logging.info("ran in {:.4f} seconds".format(end - start))

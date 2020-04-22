"""Script to calculate the sensitivity and discovery potential for CCSNe.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import sn_cats, updated_sn_catalogue_name, \
    sn_time_pdfs, raw_output_dir, raw_sn_catalogue_name
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
from flarestack.analyses.ccsn import get_sn_color
import math
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging

# Set Logger Level

logging.getLogger().setLevel("DEBUG")

logging.debug('logging level is DEBUG')

logging.getLogger('matplotlib').setLevel('INFO')

# LLH Energy PDF

llh_energy = {
    "energy_pdf_name": "power_law",
}

mh = 'fit_weights'
pdf_type = 'box'
nearby = False
nearby_str = 'nearby' if nearby else 'complte_catalogue'
cluster = 200

# Spectral indices to loop over

# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
gammas = [2.0]


# Base name

raw = raw_output_dir + "/" + mh + "/" + pdf_type + "/" + nearby_str + "/"

full_res = dict()

job_ids = []
# Loop over SN catalogues

for cat in sn_cats:

    name = raw + cat + "/"

    cat_path = updated_sn_catalogue_name(cat, nearby=nearby)
    logging.debug('catalogue path: ' + str(cat_path))
    catalogue = np.load(cat_path)

    logging.debug('catalogue dtype: ' + str(catalogue.dtype))

    closest_src = np.sort(catalogue, order="distance_mpc")[0]

    cat_res = dict()

    time_pdfs = sn_time_pdfs(cat, pdf_type=pdf_type)

    # input('stop?')

    # Loop over time PDFs

    for llh_time in time_pdfs:

        time_key = str(llh_time["post_window"] + llh_time["pre_window"])

        time_name = name + time_key + "/"

        time_res = dict()

        llh_dict = {
            "llh_name": "standard",
            "llh_energy_pdf": llh_energy,
            "llh_sig_time_pdf": llh_time,
            "llh_bkg_time_pdf": {
                "time_pdf_name": "steady"
            }
        }

        injection_time = llh_time

        # Loop over spectral indices

        for gamma in gammas:

            full_name = time_name + str(gamma) + "/"

            length = float(time_key)

            scale = (flux_to_k(reference_sensitivity(
                np.sin(closest_src["dec_rad"]), gamma=gamma
            ) * 40 * math.sqrt(float(len(catalogue)))) * 200.) / length / 4

            injection_energy = dict(llh_energy)
            injection_energy["gamma"] = gamma

            inj_dict = {
                "injection_energy_pdf": injection_energy,
                "injection_sig_time_pdf": injection_time,
                "poisson_smear_bool": True,
            }

            mh_dict = {
                "name": full_name,
                "mh_name": mh,
                "dataset": custom_dataset(ps_v002_p01, catalogue,
                                          llh_dict["llh_sig_time_pdf"]),
                "catalogue": cat_path,
                "inj_dict": inj_dict,
                "llh_dict": llh_dict,
                "scale": scale,
                "n_trials": 1000 if not cluster else 1000/cluster,
                "n_steps": 10
            }

            job_id = None
            job_id = analyse(mh_dict, cluster=cluster, n_cpu=32 if not cluster else 1, n_jobs=cluster)
            job_ids.append(job_id)

            time_res[gamma] = mh_dict

        cat_res[time_key] = time_res

    full_res[cat] = cat_res

# Wait for cluster. If there are no cluster jobs, this just runs
if cluster and np.any(job_ids):
    logging.info(f'waiting for jobs {job_ids}')
    wait_for_cluster(job_ids)

stacked_sens = dict()

for b, (cat_name, cat_res) in enumerate(full_res.items()):

    for (time_key, time_res) in cat_res.items():

        sens_livetime = []
        fracs = []
        disc_pots_livetime = []
        sens_e = []
        disc_e = []

        labels = []

        # Loop over gamma

        for (gamma, rh_dict) in sorted(time_res.items()):

            logging.debug('gamma is ' + str(gamma))

            rh = ResultsHandler(rh_dict)

            inj = rh_dict["inj_dict"]["injection_sig_time_pdf"]
            injection_length = inj["pre_window"] + inj["post_window"]

            inj_time = injection_length * 60 * 60 * 24

            # Convert IceCube numbers to astrophysical quantities

            astro_sens, astro_disc = rh.astro_values(
                rh_dict["inj_dict"]["injection_energy_pdf"])

            key = "Energy Flux (GeV cm^{-2} s^{-1})"

            e_key = "Mean Luminosity (erg/s)"

            sens_livetime.append(astro_sens[key] * inj_time)
            disc_pots_livetime.append(astro_disc[key] * inj_time)

            sens_e.append(astro_sens[e_key] * inj_time)
            disc_e.append(astro_disc[e_key] * inj_time)

            fracs.append(gamma)

            if str(gamma) not in stacked_sens.keys():
                stacked_sens[str(gamma)] = dict()

            if cat_name not in stacked_sens[str(gamma)].keys():
                stacked_sens[str(gamma)][cat_name] = [[astro_sens[e_key] * inj_time, time_key]]
            else:
                stacked_sens[str(gamma)][cat_name].append([astro_sens[e_key] * inj_time, time_key])

            # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

        name = "{0}/{1}/{2}".format(raw, cat_name, time_key)

        for j, [fluence, energy] in enumerate([[sens_livetime, sens_e],
                                              [disc_pots_livetime, disc_e]]):

            logging.debug('fluence: ' + str(fluence))
            logging.debug('energy: ' + str(energy))

            plt.figure()
            ax1 = plt.subplot(111)

            ax2 = ax1.twinx()

            # cols = ["#00A6EB", "#F79646", "g", "r"]
            linestyle = ["-", "-"][j]

            ax1.plot(fracs, fluence, label=labels, linestyle=linestyle,
                     )
            ax2.plot(fracs, energy, linestyle=linestyle,
                     )

            y_label = [r"Energy Flux (GeV cm^{-2} s^{-1})",
                       r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]

            ax2.grid(True, which='both')
            ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
            ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
            ax1.set_xlabel(r"Spectral Index ($\gamma$)")
            ax1.set_yscale("log")
            ax2.set_yscale("log")

            for k, ax in enumerate([ax1, ax2]):
                y = [fluence, energy][k]
                logging.debug('y: ' + str(y))

                ax.set_ylim(0.95 * min(y),
                            1.1 * max(y))

            plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][j] +
                      " for " + cat_name + " SNe")

            plt.tight_layout()
            plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                        ["sens", "disc"][j] + "_" + cat_name + ".pdf")
            plt.close()

for gamma in gammas:

    gamma_key = str(gamma)
    fig, ax = plt.subplots()

    for cat_name in stacked_sens[gamma_key]:
        plot_arr = np.array(stacked_sens[gamma_key][cat_name], dtype='<f8')
        logging.debug(f'plot array: \n {plot_arr}')

        yerr = plot_arr[:,0]*0.1
        patch = ax.errorbar(plot_arr[:,1], plot_arr[:,0], xerr=plot_arr[:,1]*0.05,
                            yerr=yerr, uplims=True, ls='', markersize=1, capsize=0.5, color=get_sn_color(cat_name))

        ax.plot(plot_arr[:,1], plot_arr[:,0] - yerr, marker='v', color=patch.lines[0].get_c(),
                label=cat_name, ls='')

        # if plot_results == '_figureresults':
        #     res = get_figure_limits(cat_name)
        #     ax.errorbar(res['t'], res['E'], xerr=res['t'] * 0.05, yerr=res['E']*0.1,
        #                 uplims=True, ls='', markersize=1, capsize=0.5, color=patch.lines[0].get_c(), alpha=0.5)
        #     ax.plot(res['t'], res['E'] - res['E']*0.1, marker='d', color=patch.lines[0].get_c(),
        #             label=cat_name + ' previous limit', ls='', alpha=0.5)
        #
        # if plot_results == '_tableresults':
        #     cat_key = cat_name if not 'p' in cat_name else 'IIP'
        #     ekey = 'Fit' if 'fit' or 'Fit' in mh_name else 'Fix'
        #     ekey += " Energy (erg)"
        #     ax.axhline(limits[cat_key][ekey].value,
        #                ls='--', color=patch.lines[0].get_c(), label=cat_name + ' previous limit')

    ax.set_ylabel('$E^{\\nu}_{tot}$ [erg]')
    ax.set_xlabel('Box function $\Delta T$ [d]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    plt.grid()
    plt.title(f'Stacked sensitivity for $\gamma = {gamma}$')
    plt.tight_layout()

    plt.savefig(plot_output_dir(raw + f'stacked_sensitivity_gamma{gamma}_{mh}.pdf'))
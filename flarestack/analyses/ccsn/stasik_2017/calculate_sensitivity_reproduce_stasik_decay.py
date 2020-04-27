"""Script to calculate the sensitivity and discovery potential for CCSNe.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.stasik_2017.shared_ccsn import sn_catalogue_name, sn_cats, sn_time_pdfs, pdf_names, \
    limit_sens
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits, get_figure_limits, p_vals
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import sn_time_pdfs
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

logging.debug('logging level is DEBUG')

logging.getLogger('matplotlib').setLevel('INFO')

# LLH Energy PDF

llh_energy = {
    "energy_pdf_name": "power_law",
}

cluster = 50

# Spectral indices to loop over

# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
# gammas = [1.8, 2.0, 2.5]
gammas = [2.]

# Base name

mh_name = 'fit_weights'
pdf_type = 'decay'

raw = f"analyses/ccsn/stasik_2017/calculate_sensitivity/{mh_name}/{pdf_type}/"

full_res = dict()

job_ids = []
# Loop over SN catalogues

plot_results = '_figureresults'

if __name__ == '__main__':

    for cat in ['IIp', 'IIn']:

        name = raw + cat + "/"

        # cat_path = sn_catalogue_name(cat)
        # logging.debug('catalogue path: ' + str(cat_path))
        # catalogue = np.load(cat_path)
        #
        # logging.debug('catalogue dtype: ' + str(catalogue.dtype))
        #
        # closest_src = np.sort(catalogue, order="distance_mpc")[0]

        cat_res = dict()

        time_pdfs = sn_time_pdfs(cat, pdf_type=pdf_type)

        # Loop over time PDFs

        for llh_time in time_pdfs:

            pdf_time = llh_time['decay_time'] / 364.25
            pdf_name = pdf_names(pdf_type, pdf_time)
            cat_path = sn_catalogue_name(cat, pdf_name=pdf_name)
            logging.debug('catalogue path: ' + str(cat_path))

            catalogue = np.load(cat_path)

            logging.debug('catalogue dtype: ' + str(catalogue.dtype))

            closest_src = np.sort(catalogue, order="distance_mpc")[0]

            logging.debug(f'time_pdf is {llh_time}')

            time_key = str(llh_time["post_window"] + llh_time["pre_window"]) \
                if pdf_type == 'box' \
                else str(llh_time['decay_time'])

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

                scale = 0.1 * (flux_to_k(reference_sensitivity(
                    np.sin(closest_src["dec_rad"]), gamma=gamma
                ) * 40 * math.sqrt(float(len(catalogue)))) * 200.) / length

                if cat == 'IIn':
                    scale *= 1.5

                injection_energy = dict(llh_energy)
                injection_energy["gamma"] = gamma

                inj_dict = {
                    "injection_energy_pdf": injection_energy,
                    "injection_sig_time_pdf": injection_time,
                    "poisson_smear_bool": True,
                }

                mh_dict = {
                    "name": full_name,
                    "mh_name": mh_name,
                    "dataset": custom_dataset(ps_v002_p01, catalogue,
                                              llh_dict["llh_sig_time_pdf"]),
                    "catalogue": cat_path,
                    "inj_dict": inj_dict,
                    "llh_dict": llh_dict,
                    "scale": scale,
                    "n_trials": 500/cluster if cluster else 1000,
                    "n_steps": 10
                }
                # !!! number of total trial is ntrials * n_jobs !!!

                job_id = None
                # job_id = analyse(
                #     mh_dict,
                #     cluster=True if cluster else False,
                #     n_cpu=1 if cluster else 32,
                #     n_jobs=cluster,
                #     h_cpu='02:59:59'
                # )
                job_ids.append(job_id)

                time_res[gamma] = mh_dict

            cat_res[time_key] = time_res

        full_res[cat] = cat_res

    # Wait for cluster. If there are no cluster jobs, this just runs
    if cluster and np.any(job_ids):
        logging.info(f'waiting for jobs {job_ids}')
        wait_for_cluster(job_ids)

    stacked_sens = {}
    stacked_sens_flux = {}

    for b, (cat_name, cat_res) in enumerate(full_res.items()):

        stacked_sens_flux[cat_name] = {}

        for (time_key, time_res) in cat_res.items():

            stacked_sens_flux[cat_name][time_key] = {}

            sens_livetime = []
            fracs = []
            disc_pots_livetime = []
            sens_e = []
            disc_e = []

            labels = []

            # Loop over gamma

            for (gamma, rh_dict) in sorted(time_res.items()):

                logging.debug(f'gamma is {gamma}, cat is {cat_name}, pdf time is {time_key}')

                rh = ResultsHandler(rh_dict)

                inj = rh_dict["inj_dict"]["injection_sig_time_pdf"]
                injection_length = inj["decay_length"]

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

                # if cat_name not in stacked_sens.keys():
                #     stacked_sens[cat_name] = [[astro_sens[e_key] * inj_time, time_key]]
                # else:
                #     stacked_sens[cat_name].append([astro_sens[e_key] * inj_time, time_key])

                stacked_sens_flux[cat_name][time_key][gamma] = (
                    rh.sensitivity,
                    rh.sensitivity_err,
                    astro_sens[e_key] * inj_time
                )

                # rh.ts_distribution_evolution()
                # rh.ts_evolution_gif()

                # stacked_sens_flux[cat_name][time_key][gamma] = (rh.sensitivity, rh.sensitivity_err)

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

                    # ax.set_ylim(0.95 * min(y),
                    #             1.1 * max(y))

                plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][j] +
                          " for " + cat_name + " SNe")

                plt.tight_layout()
                plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                            ["sens", "disc"][j] + "_" + cat_name + ".pdf")
                plt.close()

    # ========================      save calculated sensitivities    ======================== #

    with open(limit_sens(mh_name, pdf_type), 'wb') as f:
        pickle.dump(stacked_sens_flux, f)

    # ========================       make final plots       =============================== #

    for gamma in gammas:

        flux_fig, flux_ax = plt.subplots()

        for cat_name, cat_dict in stacked_sens_flux.items():

            x_raw = [float(key) for key in cat_dict.keys()]
            x = np.array(x_raw)[np.argsort(x_raw)] / 364.25

            y = [stacked_sens_flux[cat_name][key][gamma][0] for key in cat_dict.keys()]
            y = np.array(y)[np.argsort(x_raw)]

            yerr = [stacked_sens_flux[cat_name][key][gamma][1] for key in cat_dict.keys()]
            yerr = np.array(yerr)[np.argsort(x_raw)]

            logging.debug('\n x: {0};      y: {1};       yerr: {2}'.format(x, y, yerr))

            flux_ax.errorbar(x, y, yerr=np.array(yerr).T,
                             ls='--', label=cat_name, marker='', color=get_sn_color(cat_name))

        flux_ax.set_xscale('log')
        flux_ax.set_yscale('log')
        flux_ax.set_xlabel(r'$t_{\mathrm{decay}}$ in years')
        flux_ax.set_ylabel(r'flux in GeV$^{-1}$ s$^{-1}$ cm$^{-2}$')
        flux_ax.legend()

        fname_flux = plot_output_dir(raw + f'stacked_sensitivity_flux_gamma{gamma}_{mh_name}.pdf')
        logging.debug(f'saving figure under {fname_flux}')
        flux_fig.savefig(fname_flux)

        stacked_sens = {
            cat_name: np.array(
                [(stacked_sens_flux[cat_name][time_key][gamma][2], float(time_key) / 364.25)
                 for time_key in stacked_sens_flux[cat_name]], dtype='<f8'
            )
            for cat_name in stacked_sens_flux
        }

        # ----------------------------------------      limits      --------------------------------------- #

        fig, ax = plt.subplots()

        for cat_name in stacked_sens:
            plot_arr = np.array(stacked_sens[cat_name], dtype='<f8')
            logging.debug(f'plot array: \n {plot_arr}')

            yerr = plot_arr[:,0]*0.1
            patch = ax.errorbar(plot_arr[:,1], plot_arr[:,0], xerr=plot_arr[:,1]*0.05,
                                yerr=yerr, uplims=True, ls='', markersize=1, capsize=0.5, color=get_sn_color(cat_name))

            ax.plot(plot_arr[:,1], plot_arr[:,0] - yerr, marker='v', color=patch.lines[0].get_c(),
                    label=cat_name + ' new sensitivity', ls='')

            pval_mask = p_vals[pdf_type][cat_name]['pval'] >= 0.5

            if plot_results == '_figureresults':
                res = get_figure_limits(cat_name, pdf_type)
                res_sens = res[pval_mask]
                res_no_sens = res[np.invert(pval_mask)]

                for arr, marker, label in zip(
                        [res_sens, res_no_sens],
                        ['d', 'x'],
                        ['sensitivity', 'limit']
                ):
                    ax.errorbar(arr['t'], arr['E'], xerr=arr['t'] * 0.05, yerr=arr['E']*0.1,
                                uplims=True, ls='', markersize=1, capsize=0.5, color=patch.lines[0].get_c(), alpha=0.5)
                    ax.plot(arr['t'], arr['E'] - arr['E']*0.1, marker=marker, color=patch.lines[0].get_c(),
                            label=cat_name + ' Stasik ' + label, ls='', alpha=0.5)

            if plot_results == '_tableresults':
                cat_key = cat_name if not 'p' in cat_name else 'IIP'
                ekey = 'Fit' if 'fit' or 'Fit' in mh_name else 'Fix'
                ekey += " Energy (erg)"
                ax.axhline(limits[cat_key][ekey].value,
                           ls='--', color=patch.lines[0].get_c(), label=cat_name + ' previous limit')

        ax.set_ylabel('$E^{\\nu}_{tot}$ [erg]')
        ax.set_xlabel('t$_{\mathrm{decay}}$  [y]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

        plt.grid()
        plt.title(f'Stacked sensitivity for $\gamma = {gammas[0]}$')
        plt.tight_layout()

        plt.savefig(plot_output_dir(raw + f'stacked_sensitivity_gamma{gammas[0]}_{mh_name}{plot_results}.pdf'))
        plt.close()

        # --------------------------------------------      ratios       ------------------------------------------- #

        fig, ax = plt.subplots()

        for cat_name in stacked_sens:

            res_stasik = get_figure_limits(cat_name, pdf_type)
            plot_arr = np.array(stacked_sens[cat_name], dtype='<f8')

            plot_arr[:, 0][np.argsort(plot_arr[:, 1])] /= res_stasik['E'][np.argsort(res_stasik['t'])]

            logging.debug(f'plot array: \n {plot_arr}')

            pval_mask = p_vals[pdf_type][cat_name]['pval'] >= 0.5
            sens = plot_arr[pval_mask]
            no_sens = plot_arr[np.invert(pval_mask)]

            for arr, marker, label in zip(
                    [sens, no_sens],
                    ['d', 'x'],
                    ['sensitivity', 'limit']
            ):
                ax.plot(arr[:, 1], arr[:, 0], marker=marker, color=get_sn_color(cat_name),
                        label=f'{cat_name} {label}', ls='')

        ax.axhline(1, ls='--', color='k', label=f'ratio=1')

        ax.set_ylabel('$E^{\\nu}_{tot, \, Stasik \, reproduced} / E^{\\nu}_{tot, \, Stasik \, original}$')
        ax.set_xlabel('t$_{\mathrm{decay}}$  [y]')
        ax.set_xscale('log')
        ax.legend()

        plt.grid()
        plt.title(f'Stacked sensitivity ratio for $\gamma = {gamma}$\n'
                  f'\"Stasik reproduced with Flaresatck / Stasik original\"')
        plt.tight_layout()

        fname = plot_output_dir(raw + f'stacked_sensitivity_ratio_gamma{gamma}_{mh_name}.pdf')
        logging.debug(f'saving figure under {fname}')

        plt.savefig(fname)
        plt.close()

    end = time.time()

    logging.info('ran in {:.4f} seconds'.format(end - start))
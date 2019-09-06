"""Script to compare the sensitivity for each TDE catalogues as a function of
injected spectral index. Rather than the traditional flux at 1 GeV,
Sensitivities are given as the total integrated fluence across all sources,
and as the corresponding standard-candle-luminosity.
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.data.icecube.ps_tracks.ps_v003_p02 import ps_10year
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle, k_to_flux
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.analyses.agn_cores.shared_agncores import \
    agn_subset_catalogue, complete_cats, agn_catalogue_name
from flarestack.core.minimisation import MinimisationHandler

from flarestack.cluster import analyse
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import matplotlib.pyplot as plt
plt.style.use('~/scratch/phdthesis.mpltstyle')

import time

analyses = dict()

# Initialise Injectors/LLHs

llh_time = {
    "time_pdf_name": "Steady"
}

llh_energy = {
    "energy_pdf_name": "PowerLaw"
}

llh_dict = {
    "llh_name": "standard_matrix",
    "llh_time_pdf": llh_time,
    "llh_energy_pdf": llh_energy
}


# def base_name(cat_key, gamma):
#     return "analyses/agn_cores/test_completeness_10years_10steps_1trial/{0}/" \
#            "{1}/".format(cat_key, gamma)

def base_name(cat_key, gamma):
    return "analyses/agn_cores/test_completeness_10years_15steps_cluster/{0}/" \
           "{1}/".format(cat_key, gamma)


def generate_name(cat_key, n_sources, gamma):
    return base_name(cat_key, gamma) + "NrSrcs={0}/".format(n_sources)


# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
gammas = [2.0]

nr_brightest_sources = [int(x) for x in np.logspace(0, 4.0, 9)]

x = 13927
nr_brightest_sources += [int(x)]
print ("Nr of brightes sources to test: ", nr_brightest_sources)

all_res = dict()

running_time = []
for (cat_type, method) in complete_cats:

    unique_key = cat_type + "_" + method

    print(unique_key)

    gamma_dict = dict()

    for gamma_index in gammas:
        res = dict()
        for j, nr_srcs in enumerate(nr_brightest_sources):

            # Time it
            start_loop = time.time()

            cat_path = agn_subset_catalogue(cat_type, method, nr_srcs)

            print("Loading catalogue", cat_path, " with ", nr_srcs, "sources")
            catalogue = load_catalogue(cat_path)
            full_name = generate_name(unique_key, nr_srcs, gamma_index)
            print("Full name for ", nr_srcs, " sources is", full_name)

            injection_time = llh_time
            injection_energy = dict(llh_energy)
            injection_energy["gamma"] = gamma_index

            inj_kwargs = {
                "injection_energy_pdf": injection_energy,
                "injection_time_pdf": injection_time,
            }

            mh_dict = {
                "name": full_name,
                "mh_name": "large_catalogue",
                "datasets": ps_10year,
                "catalogue": cat_path,
                "llh_dict": llh_dict,
                "inj_dict": inj_kwargs,
                "n_trials": 1, #10,
                "n_steps": 15,
            }

            mh = MinimisationHandler.create(mh_dict)
            # mh_dict["scale"] = 2 * mh.guess_scale()
            mh_dict["scale"] = 2 * mh.guess_scale()
            # mh.trial_function()

            # pkl_file = make_analysis_pickle(mh_dict)
            analyse(mh_dict, cluster=False, n_cpu=30)


            # Check Nr. of iterations
            # niter = []
            # for i in range(10):
            #    res = mh.run_trial(0.0)
            #    niter.append(res["res"]["nfev"])
            # niter = np.array(niter)
            # print( "For ", nr_srcs, " sources, ", np.mean(niter), " nr of iterations")
            # res = mh.run_trial(0.0)
            # print( "For ", nr_srcs, " sources, ", res["res"]["nfev"])
            # print("  ")

            res[nr_srcs] = mh_dict

            end_loop = time.time()
            print("  ")
            print("RUNNING TIME using ", mh_dict["datasets"] ," catalogue and  %d sources is %s seconds " % (nr_srcs, end_loop - start_loop))
            print("  ")

        gamma_dict[gamma_index] = res
    all_res[unique_key] = gamma_dict


for (cat_key, gamma_dict) in all_res.items():

    agn_type, xray_cat = cat_key.split("_")

    full_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))

    full_flux = np.sum(full_cat["base_weight"])

    saturate_ratio = 0.26

    for (gamma_index, gamma_res) in (iter(gamma_dict.items())):

        print("gamma: ", gamma_index)
        sens_livetime = []
        n_src = []
        fracs = []
        disc_pots_livetime = []
        ratio_sens = []
        ratio_disc = []
        int_xray_flux_erg = []
        int_xray_flux = []
        guess = []

        base_dir = base_name(cat_key, gamma_index)

        for (nr_srcs, rh_dict) in sorted(gamma_res.items()):

            cat = load_catalogue(rh_dict["catalogue"])

            print("nr_srcs in loop: ", nr_srcs)
            print("   ")
            print("   ")
            int_xray = np.sum(cat["base_weight"] / 1e13*624.151)
            int_xray_flux.append(int_xray) # GeV cm-2 s-1
            int_xray_flux_erg.append(np.sum(cat["base_weight"]) / 1e13) # erg
            # cm-2 s-1
            fracs.append(np.sum(cat["base_weight"])/full_flux)

            try:
                rh = ResultsHandler(rh_dict)
                print("Sens", rh.sensitivity)
                print("Disc", rh.disc_potential)
                print("Guess", rh_dict["scale"])

                # guess.append(k_to_flux(rh_dict["scale"])* 2./3.)
                guess.append(k_to_flux(rh_dict["scale"])/3.)

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj_dict"]["injection_energy_pdf"])

                key = "Total Fluence (GeV cm^{-2} s^{-1})"

                sens_livetime.append(astro_sens[key])   # fluence=integrated over energy
                disc_pots_livetime.append(astro_disc[key])

                ratio_sens.append(astro_sens[key] / int_xray) # fluence
                # normalized over tot xray flux
                ratio_disc.append(astro_disc[key] / int_xray)

                n_src.append(nr_srcs)

            except OSError:
                pass


        labels = ['Sensitivity', 'Discovery Potential', 'sens', 'dp']
        for i, sens_dp in enumerate([sens_livetime, disc_pots_livetime]):
            print(i, sens_dp)
            # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(n_src, sens_dp, ls='-', marker='o', label = labels[i])
            ax1.grid(True, which='both')
            ax1.set_ylabel(r"Total $nu$ flux [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
            ax1.set_xlabel(r"Number of sources", fontsize=12)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] + "_vs_Nsrcs.pdf")
            plt.close()

        labels = ['Sensitivity', 'Discovery Potential', 'sens', 'dp']
        for i, sens_dp in enumerate([sens_livetime, disc_pots_livetime]):
            print(i, sens_dp)
            # Plot 2: sensitivity/dp fluence vs fraction of flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(fracs, sens_dp, ls='-', marker='o', label = labels[i])
            ax1.grid(True, which='both')
            ax1.set_ylabel(r"Total $nu$ flux [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
            ax1.set_xlabel(r"Fraction of total X-Ray flux", fontsize=12)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlim(1e-2, 1)
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] + "_vs_frac.pdf")
            plt.close()

        for i, sens_dp in enumerate([sens_livetime, disc_pots_livetime]):
            # Plot 3: sensitivity fluence vs integrated x-ray flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(int_xray_flux_erg, sens_dp, marker = 'o', ls='-',  label=labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
            ax1.set_xlabel(r"Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]",  fontsize=12)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] + "_vs_Xray_cut.pdf")
            plt.close()

        for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
            print(i, sens_dp)
            # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(n_src, sens_dp, marker ='o', ls='', label ='Ratio ' + labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(r"Integrated $\nu$ / X-Ray flux ratio", fontsize=12)
            ax1.set_xlabel(r"Number of brightest sources",  fontsize=12)
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] +
                        "_vs_Nsrcs_ratio.pdf")
            plt.close()

        for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
            print(i, sens_dp)
            # Plot 2: sensitivity/dp fluence vs fraction of total flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(fracs, sens_dp, marker ='o', ls='',
                     label='Ratio ' + labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(r"Integrated $\nu$ / X-Ray flux ratio", fontsize=12)
            ax1.set_xlabel(r"Fraction of total X-Ray flux", fontsize=12)
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
            ax1.set_xlim(1e-2, 1)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] +
                        "_vs_frac_ratio.pdf")
            plt.close()

        for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
            # Plot 3: sensitivity fluence vs integrated x-ray flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(int_xray_flux_erg, sens_dp, ls='', marker='o', label='Ratio ' + labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(r"Integrated $\nu$ / X-Ray flux ratio", fontsize=20)
            # ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
            ax1.set_xlabel(r"Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]",  fontsize=12)
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] +
                        "_vs_Xray_cut_ratio.pdf")
            plt.close()

        for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
            print(i, sens_dp)
            # Plot 2: sensitivity/dp fluence vs fraction of total flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(fracs, np.array(sens_dp) / saturate_ratio, marker ='o', ls='',
                     label='Ratio ' + labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(labels[i] + r" (cumulative population $\nu$ flux/diffuse $\nu$ flux)", fontsize=10)
            ax1.set_xlabel(r"Fraction of total X-Ray flux", fontsize=10)
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
            ax1.set_ylim(0,3)
            ax1.set_xlim(1e-2, 1)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            import matplotlib.ticker as ticker
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] +
                        "_vs_frac_diffuse.pdf")
            plt.close()

        for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
            # Plot 3: sensitivity fluence vs integrated x-ray flux
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            cols = ["#00A6EB", "#F79646", "g", "r"]
            ax1.plot(int_xray_flux_erg, np.array(sens_dp) / saturate_ratio, ls='', marker='o', label='Ratio ' + labels[i])
            ax1.grid(True, which='both')
            # ax1.legend(loc='upper left', framealpha=1)
            ax1.set_ylabel(labels[i] + r" (cumulative population $\nu$ flux/diffuse $\nu$ flux)", fontsize=12)
            ax1.set_xlabel(r"Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]",  fontsize=12)
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + labels[i+2] +
                        "_vs_Xray_cut_diffuse.pdf")
            plt.close()

        labels = ['Number of Sources', 'Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]', 'Nsrcs', 'Xray_cut_ratio']
        for i, xaxis in enumerate([n_src, int_xray_flux_erg]):
            print(i, sens_dp)
            # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(xaxis, guess, ls='-', marker='o', label = labels[i])
            ax1.grid(True, which='both')
            ax1.set_ylabel(r"Estimated Discovery Potential [GeV sr$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=12)
            ax1.set_xlabel(labels[i], fontsize=12)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + "guess_vs_" +labels[i+2] +".pdf")
            plt.close()

        labels = ['Number of Sources', 'Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]', 'Nsrcs', 'Xray_cut_ratio']
        for i, xaxis in enumerate([n_src, int_xray_flux_erg]):
            print(i, sens_dp)
            # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(xaxis, np.array(guess)/np.array(disc_pots_livetime), ls='-', marker='o', label = labels[i])
            ax1.grid(True, which='both')
            ax1.set_ylabel(r"Guess / Discovery Potential ", fontsize=12)
            ax1.set_xlabel(labels[i], fontsize=12)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + "ratio_guess_dp_vs_" +labels[i+2] +".pdf")
            plt.close()

        labels = ['Number of Sources', 'Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]', 'Nsrcs', 'Xray_cut_ratio']
        for i, xaxis in enumerate([n_src, int_xray_flux_erg]):
            print(i, sens_dp)
            # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
            plt.rc('axes', axisbelow=True)
            plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(xaxis, np.array(guess)/np.array(sens_livetime), ls='-', marker='o', label = labels[i])
            ax1.grid(True, which='both')
            ax1.set_ylabel(r"Guess / Sensitivity", fontsize=12)
            ax1.set_xlabel(labels[i], fontsize=12)
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            plt.tight_layout()
            plt.savefig(plot_output_dir(base_dir) + "ratio_guess_sen_vs_" +labels[i+2] +".pdf")
            plt.close()

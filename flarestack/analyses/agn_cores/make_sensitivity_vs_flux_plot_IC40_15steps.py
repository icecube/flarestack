"""Script to compare the sensitivity for each TDE catalogues as a function of
injected spectral index. Rather than the traditional flux at 1 GeV,
Sensitivities are given as the total integrated fluence across all sources,
and as the corresponding standard-candle-luminosity.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v003_p02 import ps_10year
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.agn_cores.shared_agncores import agn_catalogue_name
from flarestack.core.minimisation import MinimisationHandler
from flarestack.analyses.tde.shared_TDE import tde_catalogues, \
    tde_catalogue_name

from flarestack.cluster import run_desy_cluster as rd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from flarestack.utils.custom_seasons import custom_dataset

analyses = dict()

# Initialise Injectors/LLHs

llh_dict = {
    "llh_name": "standard_matrix", #"standard", #""standard_matrix",
    "LLH Time PDF": {
        "Name": "Steady"
        },
    "LLH Energy PDF":  {
        "Name": "Power Law"
        }
    }

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "Steady"
}

# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
gammas = [2.0]

nr_brightest_sources = [10, 100, 1000, 1500, 2000, 2500, 5000, 10000, 12000, 13927]
scale_per_catalogue = [1, 2, 50, 80, 100, 160, 300, 860, 900, 1000]

gamma_dict = dict()

for gamma in gammas:
    raw = "analyses/agn_cores/compare_nr_of_sources_IC40_steps15_scale/" + str(gamma) + "/"
    print "Raw name for gamma", gamma, "is", raw
    res = dict()

    for j, nr_srcs in enumerate(nr_brightest_sources):
        cat_path = agn_catalogue_name("radioloud", "radioselected_" + str(nr_srcs) + "brightest_srcs")
        catalogue = np.load(cat_path)
        print "Loading catalogue", cat_path, " with ", len(catalogue), "sources"

        full_name = raw + "NrSrcs=" + str(nr_srcs) + "/"
        print "Full name for ", nr_srcs, " sources is", full_name

        injection_time = llh_time
        injection_energy = dict(llh_energy)
        injection_energy["Gamma"] = gamma

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
        }

        scale = flux_to_k(reference_sensitivity(0.5, gamma) * 0.05*scale_per_catalogue[j])
        mh_dict = {
            "name": full_name,
            "mh_name": "large_catalogue",
            "datasets": ps_10year[:1],
            "catalogue": cat_path,
            "llh_dict": llh_dict,
            "inj kwargs": inj_kwargs,
            "n_trials": 1,
            "n_steps": 15,
        }

        mh_dict["scale"] = scale
        pkl_file = make_analysis_pickle(mh_dict)
        # To Run on the cluster

        # rd.submit_to_cluster(pkl_file, n_jobs=5)  # fast way: 1000

        # # To run locally
        # mh = MinimisationHandler.create(mh_dict)
        # mh.iterate_run(scale=scale, n_steps=mh_dict["n_steps"], n_trials=mh_dict["n_trials"])
        # mh.clear()

        res[cat_path] = mh_dict
        # res[gamma] = mh_dict
    gamma_dict[gamma] = res

rd.wait_for_cluster()

for (gamma, gamma_res) in (gamma_dict.iteritems()):

    print "gamma: ", gamma
    sens_livetime = []
    fracs = []
    disc_pots_livetime = []
    ratio_sens = []
    ratio_disc = []
    int_xray_flux_erg = []
    int_xray_flux = []

    name = "analyses/agn_cores/compare_nr_of_sources_IC40_steps15_scale/" + str(gamma) + "/"

    for (cat_path, rh_dict) in (sorted(gamma_res.iteritems())):
        # sens_livetime = [[] for _ in rh_dict]
        # fracs = [[] for _ in rh_dict]
        # disc_pots_livetime = [[] for _ in rh_dict]
        # ratio_sens = [[] for _ in rh_dict]
        # ratio_disc = [[] for _ in rh_dict]

        cat = np.load(cat_path)
        nr_srcs = len(cat)

        print "nr_srcs in loop: ", nr_srcs
        print "   "
        print "   "
        name_steps = "analyses/agn_cores/compare_nr_of_sources_IC40_steps15_scale/" + str(gamma) + "/" + "NrSrcs=" + str(nr_srcs) + "/"

        int_xray = np.sum(cat["base_weight"]/1e13*624.151)   # in GeV cm-2 s-1
        int_xray_flux.append(int_xray) # GeV cm-2 s-1
        int_xray_flux_erg.append(np.sum(cat["base_weight"]/1e13)) # erg  cm-2 s-1

        try:
            rh = ResultsHandler(rh_dict)

            astro_sens, astro_disc = rh.astro_values(
                rh_dict["inj kwargs"]["Injection Energy PDF"])

            key = "Total Fluence (GeV cm^{-2} s^{-1})"

            sens_livetime.append(astro_sens[key])   # fluence=integrated over energy
            disc_pots_livetime.append(astro_disc[key])

            ratio_sens.append(astro_sens[key]/int_xray)  # fluence normalized over tot xray flux
            ratio_disc.append(astro_disc[key]/int_xray)

            fracs.append(nr_srcs)
            print fracs

        except OSError:
            pass

        # print ratio_sens, len(ratio_sens), ratio_disc, len(ratio_disc), fracs

        # plt.figure()
        # ax1 = plt.subplot(111)
        #
        # ax2 = ax1.twinx()
        #
        # cols = ["#00A6EB", "#F79646", "g", "r"]
        # ax1.plot(fracs, sens_livetime)
        # ax2.plot(fracs, disc_pots_livetime)
        # ax2.grid(True, which='both')
        # ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        # ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
        # ax1.set_xlabel(r"Spectral Index ($\gamma$)")
        # ax1.set_yscale("log")
        # ax2.set_yscale("log")
        #
        # plt.tight_layout()
        # plt.savefig(plot_output_dir(name_steps) + "/spectral_index_" + str(gamma) +  ".pdf")
        # plt.close()

    labels = ['Sensitivity', 'Discovery Potential', 'sens', 'dp']
    for i, sens_dp in enumerate([sens_livetime, disc_pots_livetime]):
        print i, sens_dp
        # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
        plt.rc('axes', axisbelow=True)
        plt.figure()
        ax1 = plt.subplot(111)
        cols = ["#00A6EB", "#F79646", "g", "r"]
        ax1.plot(fracs, sens_dp, ls='', marker='o', label = labels[i])
        ax1.grid(True, which='both')
        # ax1.legend(loc='upper left', framealpha=1)
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
        ax1.set_xlabel(r"Number of brightest sources", fontsize=12)
        # ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/spectral_index_" + str(gamma) + "_nrsrcs_"+labels[i+2]+".pdf")
        plt.close()

    for i, sens_dp in enumerate([sens_livetime, disc_pots_livetime]):
        # Plot 3: sensitivity fluence vs integrated x-ray flux
        plt.rc('axes', axisbelow=True)
        plt.figure()
        ax1 = plt.subplot(111)
        cols = ["#00A6EB", "#F79646", "g", "r"]
        ax1.plot(int_xray_flux_erg, sens_dp, marker = 'o', ls='',  label=labels[i])
        ax1.grid(True, which='both')
        # ax1.legend(loc='upper left', framealpha=1)
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
        ax1.set_xlabel(r"Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]",  fontsize=12)
        ax1.set_xscale("log")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "spectral_index_" + str(gamma) + "_flux_"+labels[i+2]+".pdf")
        plt.close()

    for i, sens_dp in enumerate([ratio_sens, ratio_disc]):
        print i, sens_dp
        # Plot 1: sensitivity/dp fluence vs number of sources (cuts)
        plt.rc('axes', axisbelow=True)
        plt.figure()
        ax1 = plt.subplot(111)
        cols = ["#00A6EB", "#F79646", "g", "r"]
        ax1.plot(fracs, sens_dp,  marker = 'o', ls='', label = 'Ratio ' + labels[i])
        ax1.grid(True, which='both')
        # ax1.legend(loc='upper left', framealpha=1)
        ax1.set_ylabel(r"Integrated $\nu$ / X-Ray flux ratio", fontsize=12)
        ax1.set_xlabel(r"Number of brightest sources",  fontsize=12)
        # ax1.set_yscale("log")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "spectral_index_" + str(gamma) + "_nrsrcs_ratio_"+labels[i+2]+".pdf")
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
        ax1.set_ylabel(r"Integrated $\nu$ / X-Ray flux ratio", fontsize=12)
        # ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]", fontsize=12)
        ax1.set_xlabel(r"Integrated X-Ray Flux [erg cm$^{-2}$ s$^{-1}$]",  fontsize=12)
        ax1.set_xscale("log")
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "spectral_index_" + str(gamma) + "_flux_ratio_"+labels[i+2]+".pdf")
        plt.close()


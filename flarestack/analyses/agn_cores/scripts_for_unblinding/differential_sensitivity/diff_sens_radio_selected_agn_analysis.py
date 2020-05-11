"""Script to compare the sensitivity and discovery potential for the Radio-selected AGN sample (9749 sources)
as a function of injected spectral index for energy decades between 100 GeV and 10 PeV.
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from flarestack.core.results import ResultsHandler
# # from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
# from flarestack.data.icecube.ps_tracks.ps_v003_p02 import ps_10year
# from flarestack.data.icecube.northern_tracks.nt_v002_p05 import diffuse_8year
# from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle, k_to_flux

from flarestack.data.icecube import diffuse_8_year
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.analyses.agn_cores.shared_agncores import \
    agn_subset_catalogue, complete_cats_north, complete_cats_north, agn_catalogue_name, agn_subset_catalogue_north
from flarestack.core.minimisation import MinimisationHandler

from flarestack.cluster import analyse, wait_for_cluster
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# plt.style.use('~/scratch/phdthesis.mpltstyle')

import time

import logging

import os
import psutil, resource  #to get memory usage info

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
    "llh_sig_time_pdf": llh_time,
    "llh_energy_pdf": llh_energy
}

def base_name(cat_key, gamma):
    return "analyses/agn_cores/stacking_analysis_8yrNTsample_diff_sens_pre_unblinding/{0}/" \
           "{1}/".format(cat_key, gamma)


def generate_name(cat_key, n_sources, gamma):
    return base_name(cat_key, gamma) + "NrSrcs={0}/".format(n_sources)


gammas = [2.0, 2.5]

# Number of sources in the Radio-selected AGN sample
nr_brightest_sources = [9749]

# Energy bins
energies = np.logspace(2, 7, 6)
bins = list(zip(energies[:-1], energies[1:]))

all_res = dict()

running_time = []
for (cat_type, method) in complete_cats_north[:1]:

    unique_key = cat_type + "_" + method

    print(unique_key)

    gamma_dict = dict()

    for gamma_index in gammas:
        res = dict()
        for j, nr_srcs in enumerate(nr_brightest_sources):

            cat_path = agn_subset_catalogue(cat_type, method, nr_srcs)
            print("Loading catalogue", cat_path, " with ", nr_srcs, "sources")
            catalogue = load_catalogue(cat_path)
            cat = np.load(cat_path)
            print("Total flux is: ", cat['base_weight'].sum()*1e-13)
            full_name = generate_name(unique_key, nr_srcs, gamma_index)

            res_e_min = dict()
            # scale factor of neutrino injection, tuned for each energy bin
            scale_factor_per_decade = [0.2, 0.5, 1, 0.57, 0.29]

            for i, (e_min, e_max) in enumerate(bins[:]):
                full_name_en = full_name + 'Emin={0:.2f}'.format(e_min) + "/"

                print("Full name for ", nr_srcs, " sources is", full_name_en)

                injection_time = llh_time
                injection_energy = dict(llh_energy)
                injection_energy["gamma"] = gamma_index
                injection_energy["e_min_gev"] = e_min
                injection_energy["e_max_gev"] = e_max


                inj_kwargs = {
                    "injection_energy_pdf": injection_energy,
                    "injection_sig_time_pdf": injection_time,
                }

                mh_dict = {
                    "name": full_name_en,
                    "mh_name": "large_catalogue",
                    "dataset": diffuse_8_year.get_seasons(), #subselection_fraction=1),
                    "catalogue": cat_path,
                    "llh_dict": llh_dict,
                    "inj_dict": inj_kwargs,
                    "n_trials": 1, #10,
                    # "n_steps": 15,
                }

                mh = MinimisationHandler.create(mh_dict)
                scale_factor = 3 * mh.guess_scale()/3/7/scale_factor_per_decade[i]
                print("Scale Factor: ", scale_factor_per_decade[i], scale_factor)

                # # # # # How to run on the cluster for sources < 3162
                mh_dict["n_steps"] = 15
                mh_dict["scale"] = scale_factor
                # # analyse(mh_dict, cluster=False, n_cpu=35, n_jobs=10)
                analyse(mh_dict, cluster=False, n_cpu=32, n_jobs=150)

                # How to run on the cluster for sources > 3162
                # _n_jobs = 50
                # scale_loop = np.linspace(0, scale_factor, 15)
                # print(scale_loop)
                # for scale in scale_loop[:4]:
                #     print('Running ' + str(mh_dict["n_trials"]) + ' trials with scale ' + str(scale))
                #     mh_dict["fixed_scale"] = scale
                #     # # analyse(mh_dict, cluster=False, n_cpu=35, n_jobs=10)
                #     if scale == 0.:
                #         n_jobs = _n_jobs*10
                #     else:
                #         n_jobs = _n_jobs
                #     print("Submitting " + str(n_jobs) + " jobs")
                #     analyse(mh_dict, cluster=True, n_cpu=1, n_jobs=n_jobs)

                res_e_min[e_min] = mh_dict
            res[nr_srcs] = res_e_min

        gamma_dict[gamma_index] = res
    all_res[unique_key] = gamma_dict

# wait_for_cluster()

logging.getLogger().setLevel("INFO")

print(gamma_dict.items(), iter(gamma_dict.items()))

print(all_res.items(), iter(all_res.items()))

for (cat_key, gamma_dict) in all_res.items():

    print(cat_key, cat_key.split("_"))
    # agn_type, xray_cat = cat_key.split("_")[0]
    agn_type = cat_key.split("_")[0]
    print(agn_type)
    xray_cat = cat_key.split(str(agn_type)+'_')[-1]
    print(xray_cat)

    full_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))

    full_flux = np.sum(full_cat["base_weight"])

    saturate_ratio = 0.26

    for (gamma_index, gamma_res) in (iter(gamma_dict.items())):

        print("gamma: ", gamma_index)

        print("In if loop on gamma_index and res")
        print(gamma_index)
        print(gamma_res)

        sens = []
        sens_err_low = []
        sens_err_upp = []
        disc_pot = []
        disc_ts_threshold = []
        n_src = []
        fracs = []
        sens_livetime = []
        disc_pots_livetime = []
        sens_livetime_100GeV10PeV = []
        disc_pots_livetime_100GeV10PeV = []
        ratio_sens = []
        ratio_disc = []
        ratio_sens_100GeV10PeV = []
        ratio_disc_100GeV10PeV = []
        int_xray_flux_erg = []
        int_xray_flux = []
        guess = []
        sens_n = []
        disc_pot_n = []
        e_min_gev =  []
        e_max_gev = []

        base_dir = base_name(cat_key, gamma_index)

        for (nr_srcs, rh_dict_srcs) in sorted(gamma_res.items()):

            print("In if loop on nr_srcs and rh_dict")
            print(nr_srcs)
            print(rh_dict_srcs)
            print("nr_srcs in loop: ", nr_srcs)
            print("   ")
            print("   ")

            print("   ")
            print(type(rh_dict_srcs),  rh_dict_srcs)
            for (e_min, rh_dict) in sorted(rh_dict_srcs.items()):

                cat = load_catalogue(rh_dict["catalogue"])

                print("e_min in loop: ", e_min)
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
                    print("Sens_err", rh.sensitivity_err, rh.sensitivity_err[0], rh.sensitivity_err[1])
                    print("Disc", rh.disc_potential)
                    print("Disc_TS_threshold", rh.disc_ts_threshold)
                    # print("Guess", rh_dict["scale"])
                    print("Sens (n)", rh.sensitivity * rh.flux_to_ns)
                    print("DP (n)", rh.disc_potential * rh.flux_to_ns)
                    # # guess.append(k_to_flux(rh_dict["scale"])* 2./3.)
                    # guess.append(k_to_flux(rh_dict["scale"])/3.)
                    print(rh_dict["inj_dict"], rh_dict["inj_dict"]["injection_energy_pdf"]["e_min_gev"])

                    e_min_gev.append(rh_dict["inj_dict"]["injection_energy_pdf"]["e_min_gev"])
                    e_max_gev.append(rh_dict["inj_dict"]["injection_energy_pdf"]["e_max_gev"])

                    # sensitivity/dp normalized per flux normalization GeV-1 cm-2 s-1
                    sens.append(rh.sensitivity)
                    sens_err_low.append(rh.sensitivity_err[0])
                    sens_err_upp.append(rh.sensitivity_err[1])
                    disc_pot.append(rh.disc_potential)
                    disc_ts_threshold.append(rh.disc_ts_threshold)
                    sens_n.append(rh.sensitivity * rh.flux_to_ns)
                    disc_pot_n.append(rh.disc_potential * rh.flux_to_ns)

                    key = "Energy Flux (GeV cm^{-2} s^{-1})" # old version: "Total Fluence (GeV cm^{-2} s^{-1})"

                    astro_sens, astro_disc = rh.astro_values(
                        rh_dict["inj_dict"]["injection_energy_pdf"])
                    sens_livetime.append(astro_sens[key])   # fluence=integrated over energy
                    disc_pots_livetime.append(astro_disc[key])

                    # Nu energy flux integrated between 100GeV and 10PeV,
                    # indipendently from the e_min_gev, e_max_gev of the injection
                    rh_dict["inj_dict"]["injection_energy_pdf"]["e_min_gev"] = 100
                    rh_dict["inj_dict"]["injection_energy_pdf"]["e_max_gev"] = 1e7
                    astro_sens_100GeV10PeV, astro_disc_100GeV10PeV = rh.astro_values(
                        rh_dict["inj_dict"]["injection_energy_pdf"])
                    sens_livetime_100GeV10PeV.append(astro_sens_100GeV10PeV[key])   # fluence=integrated over energy
                    disc_pots_livetime_100GeV10PeV.append(astro_disc_100GeV10PeV[key])

                    # normalized over tot xray flux
                    ratio_sens.append(astro_sens[key] / int_xray) # fluence
                    ratio_disc.append(astro_disc[key] / int_xray)

                    ratio_sens_100GeV10PeV.append(astro_sens_100GeV10PeV[key] / int_xray) # fluence
                    ratio_disc_100GeV10PeV.append(astro_disc_100GeV10PeV[key] / int_xray)

                    n_src.append(nr_srcs)

                except OSError:
                    pass

        # # Save arrays to file
        np.savetxt(plot_output_dir(base_dir) + "data.out",
                   (np.array(n_src), np.array(int_xray_flux_erg),
                    np.array(e_min_gev), np.array(e_max_gev),
                    np.array(sens), np.array(sens_err_low), np.array(sens_err_upp),
                    np.array(disc_pot), np.array(disc_ts_threshold),
                    np.array(sens_livetime), np.array(disc_pots_livetime),
                    np.array(ratio_sens), np.array(ratio_disc),
                    np.array(ratio_sens)/saturate_ratio, np.array(ratio_disc)/saturate_ratio,
                    np.array(sens_livetime_100GeV10PeV), np.array(disc_pots_livetime_100GeV10PeV),
                    np.array(ratio_sens_100GeV10PeV), np.array(ratio_disc_100GeV10PeV),
                    np.array(ratio_sens_100GeV10PeV)/saturate_ratio, np.array(ratio_disc_100GeV10PeV)/saturate_ratio,
                    np.array(sens_n), np.array(disc_pot_n)),
                    header="n_src, int_xray_flux_erg, "
                           "e_min_gev, e_max_gev"
                           "sensitivity, sensitivity_err_lower, sensitivity_err_upper,"
                           "dp, disc_ts_threshold,"
                           "int_sensitivity, int_dp, ratio_sens, ratio_dp,"
                           "ratio_sens_saturate, ratio_dp_saturate,"
                           "int_sensitivity_100GeV10PeV, int_dp_100GeV10PeV, ratio_sens_100GeV10PeV, ratio_dp_100GeV10PeV,"
                           "ratio_sens_saturate_100GeV10PeV, ratio_dp_saturate_100GeV10PeV,"
                           "sensitivity_nr_neutrinos, dp_nr_neutrinos")


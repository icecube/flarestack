from __future__ import print_function
from __future__ import division
from builtins import str
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.core.results import ResultsHandler
from flarestack.cluster import run_desy_cluster as rd
from flarestack.shared import plot_output_dir, scale_shortener, \
    make_analysis_pickle
import matplotlib.pyplot as plt
import numpy as np

all_res = dict()

basename = "analyses/angular_error_floor/compare_feature/"

for gamma in [2.0, 3.0, 3.5]:
    gamma_name = basename + str(gamma) + "/"

    injection_energy = {
        "Name": "Power Law",
        "Gamma": gamma,
    }

    injection_time = {
        "Name": "Steady"
    }

    inj_dict = {
        "Injection Energy PDF": injection_energy,
        "Injection Time PDF": injection_time,
        "Poisson Smear?": False,
        "fixed_n": 100
    }

    # sin_decs = np.linspace(1.00, -1.00, 41)
    #
    # print sin_decs

    sin_decs = np.linspace(0.9, -0.9, 7)

    # print sin_decs

    # raw_input("prompt")

    # sin_decs = [-0.5, 0.0, 0.5]
    res_dict = dict()

    for pull_corrector in ["no_pull", "median_1d"]:
    # for pull_corrector in ["median_1d_e", ]:
        root_name = gamma_name + pull_corrector + "/"

        if "_e" in pull_corrector:
            root_key = "Dynamic Pull Corrector " + pull_corrector[-4] + "D "
        elif pull_corrector == "no_pull":
            root_key = "Base Case"
        else:
            root_key = "Static Pull Corrector " + pull_corrector[-2] + "D "

        for floor in ["no_floor"]:
            seed_name = root_name + floor + "/"

            if floor == "no_floor":
                key = root_key + " (No floor)"
            else:
                key = root_key + " (" + floor + ")"

            config_mh = []

            for sin_dec in sin_decs:

                name = seed_name + "sindec=" + '{0:.2f}'.format(sin_dec) + "/"

                llh_dict = {
                    "name": "spatial",
                    "LLH Energy PDF": injection_energy,
                    "LLH Time PDF": injection_time,
                    "pull_name": pull_corrector,
                    "floor_name": floor
                }

                # scale = flux_to_k(reference_sensitivity(sin_dec, gamma)) * 10

                mh_dict = {
                    "name": name,
                    "mh_name": "fixed_weights",
                    "datasets": [IC86_1_dict],
                    "catalogue": ps_catalogue_name(sin_dec),
                    "llh_dict": llh_dict,
                    "inj kwargs": inj_dict,
                    "n_trials": 50,
                    "n_steps": 2,
                    "scale": 1.
                }

                pkl_file = make_analysis_pickle(mh_dict)

                # rd.submit_to_cluster(pkl_file, n_jobs=50)
                #
                # mh_power_law = MinimisationHandler.create(mh_dict_power_law)
                # mh_power_law.iterate_run(n_steps=2, n_trials=10)

                # raw_input("prompt")

                config_mh.append(mh_dict)

            res_dict[key] = config_mh

        all_res[gamma] = res_dict

rd.wait_for_cluster()

for (gamma, res_dict) in all_res.items():

    gamma_name = basename + str(gamma) + "/"

    sens_dict = dict()
    med_bias_dict = dict()
    mean_bias_dict = dict()
    disc_dict = dict()

    for (config, mh_list) in res_dict.items():

        sens = []

        med_biases = []

        mean_biases = []

        disc = []

        for mh_dict in mh_list:
            rh = ResultsHandler(mh_dict)

            max_scale = scale_shortener(max([float(x) for x in list(rh.results.keys())]))
            sens.append(rh.sensitivity)
            disc.append(rh.disc_potential)

            fit = rh.results[max_scale]["Parameters"]["n_s"]
            inj = rh.inj[max_scale]["n_s"]
            med_bias = np.median(fit) / inj

            med_biases.append(med_bias)
            mean_biases.append(np.mean(fit) / inj)

        # ax1.plot(sin_decs, sens, label=config)
        sens_dict[config] = np.array(sens)
        med_bias_dict[config] = med_biases
        disc_dict[config] = np.array(disc)
        mean_bias_dict[config] = mean_biases

    for i, plot_dict in enumerate([sens_dict, disc_dict]):

        plt.figure()
        ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

        # ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
        ax1.grid(True, which='both')
        ax1.semilogy(nonposy='clip')

        ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]")
        plt.title(r'Comparison of Pull Corrections with $E^{-'
                  + str(gamma) + "}$")

        ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

        for (config, vals) in plot_dict.items():
            ax1.plot(sin_decs, vals, label=config)
            ax2.plot(sin_decs, vals / plot_dict["Base Case (No floor)"])

        ax2.set_xlabel(r"sin($\delta$)")
        ax2.set_ylabel("Ratio")

        ax1.set_xlim(xmin=-1.0, xmax=1.0)
        ax2.grid(True)
        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.subplots_adjust(hspace=0.001)
        ax1.legend()

        name = ["Sensitivity", "Discovery Potential"][i]

        savepath = plot_output_dir(gamma_name) + "comparison " + name + ".pdf"

        print("Saving to", savepath)
        plt.savefig(savepath)
        plt.close()

    for i, bias_dict in enumerate([med_bias_dict, mean_bias_dict]):
        name = ["Median Bias", "Mean Bias"][i]

        plt.figure()
        for (config, biases) in bias_dict.items():
            plt.plot(sin_decs, biases, label=config)
        plt.axhline(1.0, linestyle=":")
        plt.xlabel(r"sin($\delta$)")
        plt.ylabel(name + r" in $n_{s}$")
        plt.legend()
        plt.title(name + r' with $E^{-'+ str(gamma) + "}$")
        savepath = plot_output_dir(gamma_name) + "comparison " + name + ".pdf"
        print("Saving to", savepath)
        plt.savefig(savepath)
        plt.close()

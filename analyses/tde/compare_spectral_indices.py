import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_gfu_2point5_year import txs_sample, gfu_2point5
from data.icecube_pointsource_7_year import ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.skylab_reference import skylab_7year_sensitivity
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF
from utils.custom_seasons import custom_dataset

analyses = dict()

# Initialise Injectors/LLHs

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

fit_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    # "Fit Negative n_s?": True,
    "Fit Weights?": True
}

fixed_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Fit Weights?": False
}

fixed_weights_negative = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": False,
    "Fit Negative n_s?": True,
    "Fit Weights?": False
}

gammas = [1.8, 2.0, 2.1, 2.3, 2.5, 2.7]
gammas = [1.8, 2.0, 2.3]
# gammas = [1.8, 2.0, 2.3]
# gammas = [1.99, 2.0, 2.02]
# gammas = [2.5, 2.7, 2.9]

cat_res = dict()

cats = ["gold", "jetted"]
cats = ["gold"]

for cat in cats:

    name = "analyses/tde/compare_spectral_indices/" + cat + "/"

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_res = dict()

    closest_src = np.sort(catalogue, order="Distance (Mpc)")[0]

    # lengths = [0.5 * max_window]

    for i, llh_kwargs in enumerate([fixed_weights_negative,
                                    fixed_weights,
                                    fit_weights,
                                    # flare
                                    ]):
        label = ["Fixed Weights (Negative n_s)", "Fixed Weights",
                 "Fit Weights", "Flare Search", ][i]
        f_name = ["fixed_weights_neg", "fixed_weights",
                  "fit_weights", "flare"][i]

        flare_name = name + f_name + "/"

        res = dict()

        for gamma in gammas:

            full_name = flare_name + str(gamma) + "/"

            injection_time = dict(llh_time)

            injection_energy = dict(llh_energy)
            # injection_energy["E Min"] = 10000
            injection_energy["Gamma"] = gamma

            inj_kwargs = {
                "Injection Energy PDF": injection_energy,
                "Injection Time PDF": injection_time,
                "Poisson Smear?": True,
            }

            scale = flux_to_k(skylab_7year_sensitivity(
                np.sin(closest_src["dec"]), gamma=gamma
            ) * 60)


            mh_dict = {
                "name": full_name,
                "datasets": custom_dataset(txs_sample, catalogue,
                                           llh_kwargs["LLH Time PDF"]),
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 10,
                "n_steps": 10
            }

            analysis_path = analysis_dir + full_name

            try:
                os.makedirs(analysis_path)
            except OSError:
                pass

            pkl_file = analysis_path + "dict.pkl"

            with open(pkl_file, "wb") as f:
                Pickle.dump(mh_dict, f)

            rd.submit_to_cluster(pkl_file, n_jobs=100)
            #
            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=2)
            # mh.clear()
            res[gamma] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

rd.wait_for_cluster()

for (cat_name, src_res) in cat_res.iteritems():

    name = "analyses/tde/compare_spectral_indices/" + cat_name + "/"

    sens_livetime = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots_livetime = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):
        for (gamma, rh_dict) in sorted(res.iteritems()):
            try:
                rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                    rh_dict["catalogue"])

                # The uptime can noticeably devaiate from 100%
                injection_time = rh_dict["inj kwargs"]["Injection Time PDF"]

                inj_time = 0.

                cat = np.load(rh_dict["catalogue"])

                for season in rh_dict["datasets"]:
                    time = TimePDF.create(injection_time, season)
                    inj_time += np.mean([
                        time.effective_injection_time(src) for src in cat])

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj kwargs"]["Injection Energy PDF"])

                key = "Total Fluence (GeV^{-1} cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                sens_livetime[i].append(astro_sens[key] * inj_time)
                disc_pots_livetime[i].append(astro_disc[key] * inj_time)

                sens_e[i].append(astro_sens[e_key] * inj_time)
                disc_e[i].append(astro_disc[e_key] * inj_time)

                fracs[i].append(gamma)

                print rh.sensitivity * inj_time

            except OSError:
                pass

        labels.append(f_type)
        # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    for j, [fluence, energy] in enumerate([[sens_livetime, sens_e],
                                          [disc_pots_livetime, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["r", "g", "b", "orange"]
        linestyle = ["-", "--"][j]

        for i, f in enumerate(fracs):
            ax1.plot(f, fluence[i], label=labels[i], linestyle=linestyle,
                     color=cols[i])
            ax2.plot(f, energy[i], linestyle=linestyle,
                     color=cols[i])

        y_label = [r"Total Fluence [GeV cm$^{-2}$]",
                   r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]

        ax1.grid(True, which='both')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
        ax1.set_xlabel(r"Gamma")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        for k, ax in enumerate([ax1, ax2]):
            y = [fluence, energy][k]
            ax.set_ylim(0.95 * min([min(x) for x in y]),
                         1.1 * max([max(x) for x in y]))

        plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][j] +
                  " for " + cat_name + " TDEs")

        ax1.legend(loc='upper left', fancybox=True, framealpha=1.)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                    ["sens", "disc"][j] + "_" + cat_name + ".pdf")
        plt.close()

    # for j, s in enumerate([sens_livetime, sens_e]):
    #
    #     d = [disc_pots_livetime, disc_e][j]
    #
    #     for k, y in enumerate([s, d]):
    #
    #         plt.figure()
    #         ax1 = plt.subplot(111)
    #
    #         cols = ["r", "g", "b", "orange"]
    #         linestyle = ["-", "--"][k]
    #
    #         for i, f in enumerate(fracs):
    #             plt.plot(f, y[i], label=labels[i], linestyle=linestyle,
    #                      color=cols[i])
    #
    #         label = ["", "energy"][j]
    #
    #         y_label = [r"Total Fluence [GeV cm$^{-2}$]",
    #                    r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]
    #
    #         ax1.grid(True, which='both')
    #         ax1.set_ylabel(y_label[j], fontsize=12)
    #         ax1.set_xlabel(r"Gamma")
    #         ax1.set_yscale("log")
    #         ax1.set_ylim(0.95 * min([min(x) for x in y]),
    #                      1.1 * max([max(x) for x in y]))
    #
    #         plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][k] +
    #                   " for " + cat_name + " TDEs.")
    #
    #         ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
    #         plt.tight_layout()
    #         plt.savefig(plot_output_dir(name) + "/spectral_index_" + label +
    #                     "_" + ["sens", "disc"][k] + ".pdf")
    #         plt.close()
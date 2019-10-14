from __future__ import print_function
from builtins import str
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset

analyses = dict()

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

# Standard Time Integration

standard_inj_time = {
    "Name": "Box",
    "Pre-Window": 0,
    "Post-Window": 100
}

standard_inj_kwargs = {
    "Injection Time PDF": standard_inj_time,
    "Injection Energy PDF": llh_energy,
    "Poisson Smear?": True
}

standard_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
    "Fit Weights?": False
}

standard_positive_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Fit Weights?": False
}

# Murase model with One day Injection

murase_flare_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Flare Search?": True
}

inj_time_murase = {
    "Name": "Box",
    "Pre-Window": 0,
    "Post-Window": 2.3
}

murase_flare_inj_kwargs = {
    "Injection Time PDF": inj_time_murase,
    "Injection Energy PDF": llh_energy,
    "Poisson Smear?": True
}

# Winter Model with 10 day Injection

winter_energy_pdf = {
    "Name": "Power Law",
    "Gamma": 2.0
}

winter_flare_llh = {
    "LLH Energy PDF": winter_energy_pdf,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Flare Search?": True
}

winter_flare_inj_time = {
    "Name": "Box",
    "Pre-Window": 0,
    "Post-Window": 10
}

winter_flare_injection_time = {
    "Injection Time PDF": winter_flare_inj_time,
    "Injection Energy PDF": winter_energy_pdf,
    "Poisson Smear?": True,
}

# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9]
gammas = [1.8, 2.0]
# gammas = [2.0, 2.3]
# gammas = [1.99, 2.0, 2.02]
# gammas = [2.5, 2.7, 2.9]

name_root = "analyses/tde/compare_spectral_indices_individual/"

cat_res = dict()

cats = [
    "Swift J1644+57",
    # "Swift J2058+05",
    # "ASASSN-14li",
    # "XMMSL1 J0740-85"
    # "ASASSN-15lh",
]

for j, cat in enumerate(cats):

    name = name_root + cat.replace(" ", "") + "/"

    cat_path = catalogue_dir + "TDEs/individual_TDEs/" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_res = dict()

    # lengths = [0.5 * max_window]

    for i, [inj_kwargs, llh_kwargs] in enumerate([
        [standard_inj_kwargs, standard_llh],
        [standard_inj_kwargs, standard_positive_llh],
        [winter_flare_injection_time, winter_flare_llh],
        # [murase_flare_inj_kwargs, murase_flare_llh]
                                    ]):

        label = ["Time-Integrated (Negative n_s)",
                 "Time-Integrated", "10 Day Flare",
                 "2 Day Flare"][i]
        f_name = ["negative_n_s", "positive_n_s", "flare_winter",
                  "flare_murase"][i]

        flare_name = name + f_name + "/"

        res = dict()

        for gamma in gammas:

            full_name = flare_name + str(gamma) + "/"

            scale = flux_to_k(reference_sensitivity(
                np.sin(catalogue["dec"]), gamma=gamma) * 50)

            if i > 1:
                scale *= 10**(i-1)

            inj = dict(inj_kwargs)

            inj["Injection Energy PDF"] = dict(inj["Injection Energy PDF"])

            inj["Injection Energy PDF"]["Gamma"] = gamma

            if "E Min" in list(inj["Injection Energy PDF"].keys()):
                scale *= 10

            mh_dict = {
                "name": full_name,
                "datasets": custom_dataset(txs_sample_v1, catalogue,
                                           llh_kwargs["LLH Time PDF"]),
                "catalogue": cat_path,
                "inj kwargs": inj,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 5,
                "n_steps": 10
            }

            # print scale

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
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=10)
            # mh.clear()
            res[gamma] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

rd.wait_for_cluster()

for (cat, src_res) in cat_res.items():

    name = name_root + cat.replace(" ", "") + "/"

    sens = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    for i, (f_type, res) in enumerate(sorted(src_res.items())):

        for (gamma, rh_dict) in sorted(res.items()):
            try:
                rh = ResultsHandler(rh_dict)

                inj = rh_dict["inj kwargs"]["Injection Time PDF"]

                if inj["Name"] == "Box":
                    injection_length = float(inj["Pre-Window"]) + \
                                       float(inj["Post-Window"])

                else:
                    raise Exception("Unrecognised Time PDF calculation")

                inj_time = injection_length * 60 * 60 * 24

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj kwargs"]["Injection Energy PDF"])

                key = "Total Fluence (GeV cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                sens[i].append(astro_sens[key] * inj_time)
                disc_pots[i].append(astro_disc[key] * inj_time)

                sens_e[i].append(astro_sens[e_key] * inj_time)
                disc_e[i].append(astro_disc[e_key] * inj_time)

                fracs[i].append(gamma)

            except OSError:
                pass

        labels.append(f_type)

    for j, [fluence, energy] in enumerate([[sens, sens_e],
                                          [disc_pots, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["r", "g", "b", "orange"]
        linestyle = ["-", "--"][j]

        print(fracs, fluence, labels, cols, energy)

        for l, f in enumerate(fracs):

            try:
                ax1.plot(f, fluence[l], label=labels[l], linestyle=linestyle,
                         color=cols[l])
                ax2.plot(f, energy[l], linestyle=linestyle,
                         color=cols[l])
            except ValueError:
                pass

        y_label = [r"Total Fluence [GeV cm$^{-2}$]",
                   r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]

        ax2.grid(True, which='both')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
        ax1.set_xlabel(r"Gamma")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        for k, ax in enumerate([ax1, ax2]):
            y = [fluence, energy][k]

            ax.set_ylim(0.95 * min([min(x) for x in y if len(x) > 0]),
                        1.1 * max([max(x) for x in y if len(x) > 0]))

        plt.title(["Sensitivity", "Discovery Potential"][j] + " for " + cat)

        ax1.legend(loc='upper left', fancybox=True, framealpha=1.)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                    ["sens", "disc"][j] + "_" + cat + ".pdf")
        plt.close()

    # for j, s in enumerate([sens, sens_e]):
    #
    #     d = [disc_pots, disc_e][j]
    #
    #     for k, y in enumerate([s, d]):
    #
    #         plt.figure()
    #         ax1 = plt.subplot(111)
    #
    #         cols = ["b", "orange", "green"]
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
    #         print y
    #
    #         plt.title("Time-Integrated Emission")
    #
    #         ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
    #         plt.tight_layout()
    #
    #         print label, k
    #
    #         plt.savefig(plot_output_dir(name) + "/spectral_index_" + label +
    #                     "_" + ["sens", "disc"][k] + ".pdf")
    #         plt.close()
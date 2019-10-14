from builtins import str
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.analyses.txs_0506_056.make_txs_catalogue import txs_cat_path

catalogue = np.load(txs_cat_path)

# Initialise Injectors/LLHs

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

fixed_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
}

fixed_weights_negative = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
}

gammas = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 2.7]
# gammas = [1.8, 2.0]


# power_law_start_energy = [100, 10000, 100000]
power_law_start_energy = [100]

cutoff_dict = dict()

injection_length = 100

for e_min in power_law_start_energy:

    name_root = "analyses/txs_0506_056/compare_spectral_indices/Emin=" + \
                str(e_min) + "/"

    src_res = dict()

    for i, llh_kwargs in enumerate([fixed_weights_negative,
                                    fixed_weights,
                                    # flare
                                    ]):
        label = ["Fixed Weights (Negative n_s)", "Fixed Weights",
                 "Flare Search", ][i]
        f_name = ["fixed_weights_neg", "fixed_weights",
                  "flare"][i]

        name = name_root + f_name + "/"

        res = dict()

        for gamma in gammas:

            full_name = name + str(gamma) + "/"

            injection_time = llh_time = {
                "Name": "FixedEndBox"
            }

            injection_energy = dict(llh_energy)
            injection_energy["E Min"] = e_min
            injection_energy["Gamma"] = gamma

            inj_kwargs = {
                "Injection Energy PDF": injection_energy,
                "Injection Time PDF": injection_time,
                "Poisson Smear?": True,
            }

            scale = flux_to_k(reference_sensitivity(
                np.sin(catalogue["dec"]), gamma=gamma
            )) * 60 * (e_min/100.)**0.2

            mh_dict = {
                "name": full_name,
                "datasets": custom_dataset(txs_sample_v1, catalogue,
                                           llh_kwargs["LLH Time PDF"]),
                "catalogue": txs_cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 5,
                "n_steps": 15
            }

            analysis_path = analysis_dir + full_name

            try:
                os.makedirs(analysis_path)
            except OSError:
                pass

            pkl_file = analysis_path + "dict.pkl"

            with open(pkl_file, "wb") as f:
                Pickle.dump(mh_dict, f)

            # rd.submit_to_cluster(pkl_file, n_jobs=1000)

            # mh_power_law = MinimisationHandler(mh_dict_power_law)
            # mh_power_law.iterate_run(mh_dict_power_law["scale"], mh_dict_power_law["n_steps"],
            #                n_trials=10)
            # mh_power_law.clear()

            res[gamma] = mh_dict

        src_res[label] = res

    cutoff_dict[e_min] = src_res

# rd.wait_for_cluster()

for (e_min, src_res) in cutoff_dict.items():

    name = "analyses/txs_0506_056/compare_spectral_indices/Emin=" + \
          str(e_min) + "/"

    sens_livetime = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots_livetime = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    for i, (f_type, res) in enumerate(sorted(src_res.items())):

        # if f_type == "Fit Weights":
        if True:

            for (gamma, rh_dict) in sorted(res.items()):
                try:
                    rh = ResultsHandler(rh_dict)

                    inj_time = injection_length * 60 * 60 * 24

                    astro_sens, astro_disc = rh.astro_values(
                        rh_dict["inj kwargs"]["Injection Energy PDF"])

                    key = "Total Fluence (GeV cm^{-2} s^{-1})"

                    e_key = "Mean Luminosity (erg/s)"

                    sens_livetime[i].append(astro_sens[key] * inj_time)
                    disc_pots_livetime[i].append(astro_disc[key] * inj_time)

                    sens_e[i].append(astro_sens[e_key])
                    disc_e[i].append(astro_disc[e_key])

                    fracs[i].append(gamma)

                except OSError:
                    pass

            labels.append(f_type)
        # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    for j, [fluence, energy] in enumerate([[sens_livetime, sens_e],
                                          [disc_pots_livetime, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["#00A6EB", "#F79646", "g", "r"]
        linestyle = ["-", "-"][j]

        for i, f in enumerate(fracs):

            if len(f) > 0:

                ax1.plot(f, fluence[i], label=labels[i], linestyle=linestyle,
                         color=cols[i])
                ax2.plot(f, energy[i], linestyle=linestyle,
                         color=cols[i])

        ax2.grid(True, which='both')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        ax2.set_ylabel(r"Isotropic-Equivalent Luminosity $L_{\nu}$ (ergs s$^{"
                       r"-1}$)")
        ax1.set_xlabel(r"Spectral Index ($\gamma$)")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        for k, ax in enumerate([ax1, ax2]):
            y = [fluence, energy][k]

            ax.set_ylim(0.95 * min([min(x) for x in y if len(x) > 0]),
                        1.1 * max([max(x) for x in y if len(x) > 0]))

        plt.title(["Sensitivity", "Discovery Potential"][j] +
                  " for TXS 0506+56 (Neutrino Flare)")

        ax1.legend(loc='upper left', fancybox=True, framealpha=0.)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                    "Emin=" + str(e_min) +
                    ["sens", "disc"][j] + "_TXS_0506+56.pdf")
        plt.close()


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
import math
import matplotlib as mpl
mpl.use('Agg')

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

new_style = {
    "font.family": "serif",
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
 }
mpl.rcParams.update(new_style)

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
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
    "Fit Weights?": False
}

gammas = [1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 2.7]
gammas = [1.8, 2.0, 2.3, 2.5, 2.7]
gammas = [1.8, 2.0]

# cats = ["gold", "jetted"]
# # cats = ["jetted"]
# cat_names = ["Jetted", "Golden"]
cats = ["jetted"]
cat_names = ["Jetted"]

power_law_start_energy = [100, 10000, 100000]
power_law_start_energy = [100]

cutoff_dict = dict()

injection_length = 100

for e_min in power_law_start_energy:

    raw = "analyses/tde/compare_spectral_indices/" + "Emin=" + str(e_min) + "/"

    cat_res = dict()

    for cat in cats:

        name = raw + cat + "/"

        cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
        catalogue = np.load(cat_path)

        src_res = dict()

        closest_src = np.sort(catalogue, order="Distance (Mpc)")[0]

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

                injection_time = llh_time = {
                    "Name": "Box",
                    "Pre-Window": 0.,
                    "Post-Window": injection_length
                }

                injection_energy = dict(llh_energy)
                injection_energy["E Min"] = e_min
                injection_energy["Gamma"] = gamma

                inj_kwargs = {
                    "Injection Energy PDF": injection_energy,
                    "Injection Time PDF": injection_time,
                    "Poisson Smear?": True,
                }

                scale = flux_to_k(skylab_7year_sensitivity(
                    np.sin(closest_src["dec"]), gamma=gamma
                ) * 40 * math.sqrt(float(len(catalogue)))) * (e_min/100.)**0.2

                mh_dict = {
                    "name": full_name,
                    "datasets": custom_dataset(txs_sample, catalogue,
                                               llh_kwargs["LLH Time PDF"]),
                    "catalogue": cat_path,
                    "inj kwargs": inj_kwargs,
                    "llh kwargs": llh_kwargs,
                    "scale": scale,
                    "n_trials": 20,
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

                # if label == "Fit Weights":
                rd.submit_to_cluster(pkl_file, n_jobs=500)

                # mh = MinimisationHandler(mh_dict)
                # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"],
                #                n_trials=10)
                # mh.clear()

                res[gamma] = mh_dict

            src_res[label] = res

        cat_res[cat] = src_res

    cutoff_dict[e_min] = cat_res

rd.wait_for_cluster()

for (e_min, cat_res) in cutoff_dict.iteritems():

    raw = "analyses/tde/compare_spectral_indices/" + "Emin=" + str(e_min) + "/"

    for b, (cat_name, src_res) in enumerate(cat_res.iteritems()):

        name = raw + cat_name + "/"

        sens_livetime = [[] for _ in src_res]
        fracs = [[] for _ in src_res]
        disc_pots_livetime = [[] for _ in src_res]
        sens_e = [[] for _ in src_res]
        disc_e = [[] for _ in src_res]

        labels = []

        for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):

            # if f_type == "Fit Weights":
            if True:

                for (gamma, rh_dict) in sorted(res.iteritems()):
                    try:
                        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                            rh_dict["catalogue"], show_inj=True)

                        inj_time = injection_length * 60 * 60 * 24

                        astro_sens, astro_disc = rh.astro_values(
                            rh_dict["inj kwargs"]["Injection Energy PDF"])

                        key = "Total Fluence (GeV^{-1} cm^{-2} s^{-1})"

                        e_key = "Mean Luminosity (erg/s)"

                        sens_livetime[i].append(astro_sens[key] * inj_time)
                        disc_pots_livetime[i].append(astro_disc[key] * inj_time)

                        sens_e[i].append(astro_sens[e_key] * inj_time)
                        disc_e[i].append(astro_disc[e_key] * inj_time)

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

            y_label = [r"Total Fluence [GeV cm$^{-2}$]",
                       r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]

            ax2.grid(True, which='both')
            ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
            ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
            ax1.set_xlabel(r"Spectral Index ($\gamma$)")
            ax1.set_yscale("log")
            ax2.set_yscale("log")

            for k, ax in enumerate([ax1, ax2]):
                y = [fluence, energy][k]

                ax.set_ylim(0.95 * min([min(x) for x in y if len(x) > 0]),
                            1.1 * max([max(x) for x in y if len(x) > 0]))

            plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][j] +
                      " for " + cat_names[b] + " TDEs")

            ax1.legend(loc='upper left', fancybox=True, framealpha=1.)
            plt.tight_layout()
            plt.savefig(plot_output_dir(name) + "/spectral_index_" +
                        "Emin=" + str(e_min) +
                        ["sens", "disc"][j] + "_" + cat_name + ".pdf")
            plt.close()

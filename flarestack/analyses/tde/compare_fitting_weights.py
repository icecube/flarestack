"""Script to compare the four alternative stacking methods tested for the TDE
catalogues. Compares the regular stacking, as well as stacking with negative
n_s, stacking where an n_s for each source is fit individually (rather than a
joint n_s split across sources assuming standard candles), and lastly an
independent flare search for each source with TS values summed at the end.
"""
from __future__ import division
from builtins import str
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p02 import txs_sample_v2
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import math
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name, \
    tde_catalogues

analyses = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox",
}

llh_energy = injection_energy

fit_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Weights?": True,
    "Fit Negative n_s?": False,
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

flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True,
    "Fit Negative n_s?": False
}

cat_res = dict()

# Inject for, at most, 100 days

max_window = 100

lengths = np.logspace(-2, 0, 5) * max_window

for cat in tde_catalogues:

    name = "analyses/tde/compare_fitting_weights/" + cat + "/"

    cat_path = tde_catalogue_name(cat)
    catalogue = np.load(cat_path)

    src_res = dict()

    closest_src = np.sort(catalogue, order="Distance (Mpc)")[0]

    for i, llh_kwargs in enumerate([
                                    fixed_weights,
                                    fixed_weights_negative,
                                    fit_weights,
                                    flare
                                    ]):
        label = ["Fixed Weights", "Fixed Weights (Negative n_s)",
                 "Fit Weights", "Flare Search", ][i]
        f_name = ["fixed_weights", "fixed_weights_neg",
                  "fit_weights", "flare"][i]

        flare_name = name + f_name + "/"

        res = dict()

        for flare_length in lengths:

            full_name = flare_name + str(flare_length) + "/"

            # Use a box time PDF of length flare_length to inject signal.
            # Randomly move this box within the search window, to average over
            # livetime fluctuations/detector seasons.

            injection_time = {
                "Name": "Box",
                "Pre-Window": 0,
                "Post-Window": flare_length,
                "Time Smear?": True,
                "Min Offset": 0.,
                "Max Offset": max_window - flare_length
            }

            inj_kwargs = {
                "Injection Energy PDF": injection_energy,
                "Injection Time PDF": injection_time,
                "Poisson Smear?": True,
            }

            scale = old_div(100 * math.sqrt(float(len(catalogue))) * flux_to_k(
                reference_sensitivity(np.sin(closest_src["dec"]), gamma=2)
            ) * max_window, flare_length)

            # print scale

            mh_dict = {
                "name": full_name,
                "datasets": custom_dataset(txs_sample_v2, catalogue,
                                           llh_kwargs["LLH Time PDF"]),
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 1,
                "n_steps": 15
            }

            pkl_file = make_analysis_pickle(mh_dict)

            # rd.submit_to_cluster(pkl_file, n_jobs=500)

            res[flare_length] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

# rd.wait_for_cluster()

for (cat, src_res) in cat_res.items():

    name = "analyses/tde/compare_fitting_weights/" + cat + "/"

    sens = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    cat_path = tde_catalogue_name(cat) + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_1_frac = old_div(min(catalogue["Distance (Mpc)"])**-2,np.sum(
        catalogue["Distance (Mpc)"] ** -2
    ))

    for i, (f_type, res) in enumerate(sorted(src_res.items())):

        for (length, rh_dict) in sorted(res.items()):
            try:
                rh = ResultsHandler(rh_dict)

                # Convert flux to fluence and source energy

                inj_time = length * 60 * 60 * 24

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj kwargs"]["Injection Energy PDF"])

                key = "Total Fluence (GeV cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                sens[i].append(astro_sens[key] * inj_time)
                disc_pots[i].append(astro_disc[key] * inj_time)

                sens_e[i].append(astro_sens[e_key] * inj_time)
                disc_e[i].append(astro_disc[e_key] * inj_time)

                fracs[i].append(length)

            except OSError:
                pass

        labels.append(f_type)

    for j, [fluence, energy] in enumerate([[sens, sens_e],
                                           [disc_pots, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["#F79646", "#00A6EB", "g", "r"]

        for i, f in enumerate(fracs):

            if len(f) > 0:
                # Plot fluence on left y axis, and source energy on right y axis

                ax1.plot(f, fluence[i], label=labels[i], color=cols[i])
                ax2.plot(f, energy[i], color=cols[i])

        # Set up plot

        ax2.grid(True, which='both')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
        ax1.set_xlabel(r"Flare Length (Days)")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        # Set limits and save

        for k, ax in enumerate([ax1, ax2]):

            try:
                y = [fluence, energy][k]

                ax.set_ylim(0.7 * min([min(x) for x in y if len(x) > 0]),
                            1.5 * max([max(x) for x in y if len(x) > 0]))

            except ValueError:
                pass

        plt.title(["Sensitivity", "Discovery Potential"][j] + " for " + cat +
                  " TDEs")

        ax1.legend(loc='upper left', fancybox=True)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/flare_vs_box_" + cat + "_" +
                    ["sens", "disc"][j] + ".pdf")
        plt.close()

import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p02 import txs_sample_v2
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.utils.custom_seasons import custom_dataset
import matplotlib.pyplot as plt
from flarestack.analyses.tde.shared_TDE import individual_tdes, \
    individual_tde_cat

name_root = "analyses/tde/compare_cluster_search_to_time_integration/"

cat_res = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox",
}

llh_energy = injection_energy

# A standard time integration, with n_s >=0

time_integrated = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": False
}

# Time integration where n_s can be fit as negative or positive

time_integrated_negative_n_s = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": True
}

# A flare search, looking for clustering in time and space

flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True,
    "Fit Negative n_s?": False
}

for j, cat in enumerate(individual_tdes):

    name = name_root + cat.replace(" ", "") + "/"

    cat_path = individual_tde_cat(cat)
    catalogue = np.load(cat_path)

    t_start = catalogue["Start Time (MJD)"]
    t_end = catalogue["End Time (MJD)"]

    max_window = float(t_end - t_start)

    src_res = dict()

    lengths = np.logspace(-2, 0, 5) * max_window
    # lengths = np.logspace(-2, 0, 17) * max_window

    # Loop over likelihood methods

    for i, llh_kwargs in enumerate([time_integrated,
                                    time_integrated_negative_n_s,
                                    flare
                                    ]):

        # Set plot labels and subdirectory names

        label = ["Time-Integrated", "Time-Integrated (negative n_s)",
                 "Cluster Search"][i]
        f_name = ["fixed_box", "fixed_box_negative", "flare"][i]

        flare_name = name + f_name + "/"

        res = dict()

        # Loop over different lengths for injection

        for flare_length in lengths:

            full_name = flare_name + str(flare_length) + "/"

            # Use a box time PDF of length flare_length to inject signal.
            # Randomly move this box within the search window, to average over
            # livetime fluctuations/detector seasons.

            injection_time = {
                "Name": "FixedRefBox",
                "Fixed Ref Time (MJD)": t_start,
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

            # Sets a default flux scale for signal injection

            scale = flux_to_k(reference_sensitivity(np.sin(catalogue["dec"]))
                              * (50 * max_window / flare_length))

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

            analysis_path = analysis_dir + full_name

            try:
                os.makedirs(analysis_path)
            except OSError:
                pass

            pkl_file = analysis_path + "dict.pkl"

            with open(pkl_file, "wb") as f:
                Pickle.dump(mh_dict, f)

            # Run jobs on cluster

            # rd.submit_to_cluster(pkl_file, n_jobs=5000)

            # Run locally
            #
            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], n_steps=10, n_trials=3)
            # mh.clear()

            res[flare_length] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

# Wait for cluster jobs to finish

# rd.wait_for_cluster()

for (cat, src_res) in cat_res.iteritems():

    name = name_root + cat.replace(" ", "") + "/"

    sens = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    # Loop over likelihood methods

    for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):

        # if f_type != "Time-Integrated (negative n_s)":

        for (length, rh_dict) in sorted(res.iteritems()):

            # Calculate the sensitivity based on TS distributions

            rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                rh_dict["catalogue"], show_inj=True)

            # Length of injection in seconds

            inj_time = length * (60 * 60 * 24)

            # Convert flux to fluence and source energy

            astro_sens, astro_disc = rh.astro_values(
                rh_dict["inj kwargs"]["Injection Energy PDF"])

            key = "Total Fluence (GeV cm^{-2} s^{-1})"

            e_key = "Mean Luminosity (erg/s)"

            sens[i].append(astro_sens[key] * inj_time)
            disc_pots[i].append(astro_disc[key] * inj_time)

            sens_e[i].append(astro_sens[e_key] * inj_time)
            disc_e[i].append(astro_disc[e_key] * inj_time)

            fracs[i].append(length)

        labels.append(f_type)

    # Loop over sensitivity/discovery potential

    for j, [fluence, energy] in enumerate([[sens, sens_e],
                                          [disc_pots, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["#F79646", "#00A6EB", "g", "r"]
        linestyle = ["-", "-"][j]

        for i, f in enumerate(fracs):

            if len(f) > 0:

                # Plot fluence on left y axis, and source energy on right y axis

                ax1.plot(f, fluence[i], label=labels[i], linestyle=linestyle,
                         color=cols[i])
                ax2.plot(f, energy[i], linestyle=linestyle,
                         color=cols[i])

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

        plt.title(["Sensitivity", "Discovery Potential"][j] + " for " + cat)

        ax1.legend(loc='upper left', fancybox=True)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) + "/flare_vs_box_" +
                    ["sens", "disc"][j] + ".pdf")
        plt.close()

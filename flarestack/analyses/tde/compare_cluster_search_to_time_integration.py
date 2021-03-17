"""Script to compare the standard time integration method to time integration
with negative n_s, and also to the flare search method, which looks for
temporal clustering. The script runs for all individual TDEs to be analysed.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p02 import txs_sample_v2
from flarestack.data.icecube.gfu.gfu_v002_p04 import gfu_v002_p04
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.utils import load_catalogue, custom_dataset
from flarestack.cluster import analyse, wait_cluster
import matplotlib.pyplot as plt
from flarestack.analyses.tde.shared_TDE import individual_tdes, \
    individual_tde_cat
import logging

logging.basicConfig(level=logging.INFO)

name_root = "analyses/tde/compare_cluster_search_to_time_integration/"

cat_res = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

llh_time = {
    "time_pdf_name": "custom_source_box"
}

llh_bkg_time = {
    "time_pdf_name": "steady"
}

llh_energy = injection_energy

# A standard time integration, with n_s >=0

time_integrated = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": llh_bkg_time,
    "negative_ns_bool": False
}

# Time integration where n_s can be fit as negative or positive

time_integrated_negative_n_s = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": llh_bkg_time,
    "negative_ns_bool": True
}

configs = [
    # (time_integrated, "fixed_weights", "fixed_box", "Time-Integrated (n_s > 0)"),
    (time_integrated_negative_n_s, "fixed_weights", "fixed_box_negative", "Time-Integrated"),
    (time_integrated, "flare", "flare", "Cluster Search")
]

cluster = True

job_ids = []

for j, cat in enumerate(individual_tdes[-1:]):

    name = name_root + cat.replace(" ", "") + "/"

    cat_path = individual_tde_cat(cat)
    catalogue = load_catalogue(cat_path)

    t_start = catalogue["start_time_mjd"]
    t_end = catalogue["end_time_mjd"]

    max_window = float(t_end - t_start)

    src_res = dict()

    lengths = np.logspace(-2, 0, 2) * max_window

    # Loop over likelihood methods

    for (llh_dict, mh_name, f_name, label) in configs:

        # Set plot labels and subdirectory names

        flare_name = name + f_name + "/"

        res = dict()

        # Loop over different lengths for injection

        for flare_length in lengths:

            full_name = flare_name + str(flare_length) + "/"

            # Use a box time PDF of length flare_length to inject signal.
            # Randomly move this box within the search window, to average over
            # livetime fluctuations/detector seasons.

            injection_time = {
                "time_pdf_name": "fixed_ref_box",
                "fixed_ref_time_mjd": t_start,
                "pre_window": 0.,
                "post_window": flare_length,
                "time_smear_bool": True,
                "min_offset": 0.,
                "max_offset": max_window - flare_length
            }

            inj_dict = {
                "injection_energy_pdf": injection_energy,
                "injection_sig_time_pdf": injection_time,
            }

            # Sets a default flux scale for signal injection

            scale = flux_to_k(reference_sensitivity(np.sin(catalogue["dec_rad"]))
                              * (50 * max_window / flare_length))

            if cat != "AT2018cow":
                dataset = custom_dataset(txs_sample_v2, catalogue,
                                         llh_dict["llh_sig_time_pdf"])
            else:
                dataset = gfu_v002_p04

            mh_dict = {
                "name": full_name,
                "mh_name": mh_name,
                "dataset": dataset,
                "catalogue": cat_path,
                "inj_dict": inj_dict,
                "llh_dict": llh_dict,
                "scale": scale,
                "n_trials": 10,
                "n_steps": 15
            }

            # Run jobs on cluster

            job_id = analyse(mh_dict,
                             cluster=cluster,
                             n_cpu=1 if cluster else 32,
                             h_cpu='00:59:59')
            job_ids.append(job_id)

            res[flare_length] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

wait_cluster(job_ids)

# Wait for cluster jobs to finish

for (cat, src_res) in cat_res.items():

    name = name_root + cat.replace(" ", "") + "/"

    sens = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    # Loop over likelihood methods

    for i, (f_type, res) in enumerate(sorted(src_res.items())):

        if f_type != "Time-Integrated (n_s > 0)":

            for (length, rh_dict) in sorted(res.items()):

                # Calculate the sensitivity based on TS distributions

                rh = ResultsHandler(rh_dict)

                # Length of injection in seconds

                inj_time = length * (60 * 60 * 24)

                # Convert flux to fluence and source energy

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj_dict"]["injection_energy_pdf"])

                key = "Energy Flux (GeV cm^{-2} s^{-1})"

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

        path = plot_output_dir(name) + "/flare_vs_box_" + ["sens", "disc"][j] + ".pdf"

        logging.info(f"Saving to {path}")

        plt.savefig(path)
        plt.close()

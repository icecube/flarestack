"""
Script to reproduce the analysis of the 1ES 1959+650 blazar, as described in
https://wiki.icecube.wisc.edu/index.php/1ES_1959_Analysis.

The script can be used to verify that the flare search method, as implemented
here, is capable of matching previous flare search methods.
"""
from __future__ import print_function
from __future__ import division
from builtins import str
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import txs_sample_v1
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.shared import plot_output_dir, flux_to_k, transients_dir
from flarestack.utils.prepare_catalogue import custom_sources
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import analyse, wait_for_cluster
import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import Distance

name = "analyses/fermi/GB6_blazar_flare_search/"

analyses = dict()

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

# Start and end time of flare in MJD
t_start = 55753.00
t_end = 56000.00
# t_end = 56474.00 #true end hard flare

# Ra and dec of source
ra = 160.134167
dec = 6.29

# Distance to source -> z=0.048 according to TeVCat. With lambdaCDM, this gives:
z = 0.73
lumdist = Distance(z=z).to("Mpc").value

# Creates the .npy source catalogue
catalogue = custom_sources(
    name="GB6_J1040_0617",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=lumdist,
    start_time=t_start,
    end_time=t_end,
)

cat_path = transients_dir + "GB6_J1040_0617.npy"
np.save(cat_path, catalogue)


search_window = float(t_end - t_start)
max_window = 150.
# Initialise Injectors/LLHs

injection_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": 2.0,
}

llh_time = {
    "time_pdf_name": "FixedRefBox",
    "fixed_ref_time_mjd": t_start,
    "pre_window": 0.,
    "post_window": search_window,
    "max_flare": max_window
}

llh_energy = injection_energy

no_flare = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_time_pdf": llh_time,
}

no_flare_negative = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_time_pdf": llh_time,
    "negative_ns_bool": True
}

flare = no_flare

src_res = dict()

lengths = np.logspace(-2, 0, 2) * max_window

for i, llh_kwargs in enumerate([
                                no_flare,
                                no_flare_negative,
                                flare
                                ]):

    label = ["Time-Independent", "Time-Independent (negative n_s)",
             "Time-Clustering"][i]
    f_name = ["fixed_box", "fixed_box_negative", "flare_fit_gamma"][i]
    mh_name = ["fixed_weights", "fixed_weights", "flare"][i]

    flare_name = name + f_name + "/"

    res = dict()

    for flare_length in lengths:

        full_name = flare_name + str(flare_length) + "/"

        injection_time = {
            "time_pdf_name": "FixedRefBox",
            "fixed_ref_time_mjd": t_start,
            "pre_window": 0,
            "post_window": flare_length,
            "time_smear_bool": True,
            "min_offset": 0.,
            "max_offset": search_window - flare_length
        }

        inj_kwargs = {
            "injection_energy_pdf": injection_energy,
            "injection_time_pdf": injection_time,
        }

        scale = flux_to_k(reference_sensitivity(np.sin(dec))
                          * (50 * search_window/ flare_length))

        mh_dict = {
            "name": full_name,
            "mh_name": mh_name,
            "datasets": custom_dataset(txs_sample_v1, catalogue,
                                       llh_kwargs["llh_time_pdf"]),
            "catalogue": cat_path,
            "inj_dict": inj_kwargs,
            "llh_dict": llh_kwargs,
            "scale": scale,
            "n_trials": 100,
            "n_steps": 15 #number of flux values
        }

        # if mh_name == "flare":
            
        analyse(mh_dict, n_cpu=24, cluster=False)

        # raw_input("prompt")

        res[flare_length] = mh_dict

    src_res[label] = res

wait_for_cluster() #for cluster

sens = [[] for _ in src_res]
fracs = [[] for _ in src_res]
disc_pots = [[] for _ in src_res]
sens_e = [[] for _ in src_res]
disc_e = [[] for _ in src_res]

labels = []

for i, (f_type, res) in enumerate(sorted(src_res.items())):
    for (length, rh_dict) in sorted(res.items()):

        rh = ResultsHandler(rh_dict)

        inj_time = length * (60 * 60 * 24)

        astro_sens, astro_disc = rh.astro_values(
            rh_dict["inj_dict"]["injection_energy_pdf"])

        e_key = "Mean Luminosity (erg/s)"

        sens[i].append(rh.sensitivity * inj_time)
        disc_pots[i].append(rh.disc_potential * inj_time)
        sens_e[i].append(astro_sens[e_key] * inj_time)
        disc_e[i].append(astro_disc[e_key] * inj_time)
        fracs[i].append(length)

    labels.append(f_type)
    # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

for j, [fluence, energy] in enumerate([[sens, sens_e],
                                       [disc_pots, disc_e]]):

    plt.figure()
    ax1 = plt.subplot(111)

    ax2 = ax1.twinx()

    cols = ["#F79646","#00A6EB",  "g", "r"]
    linestyle = ["-", "-"][j]

    for i, f in enumerate(fracs):

        print(fluence[i], labels[i])

        ax1.plot(f, fluence[i], label=labels[i], linestyle=linestyle,
                 color=cols[i])
        ax2.plot(f, energy[i], linestyle=linestyle,
                 color=cols[i])

    ax1.grid(True, which='both')
    ax1.set_ylabel(r"Time-Integrated Flux[ GeV$^{-1}$ cm$^{-2}$]",
                   fontsize=12)
    ax2.set_ylabel(r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)")
    ax1.set_xlabel(r"Flare Length (days)")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    for k, ax in enumerate([ax1, ax2]):
        y = [fluence, energy][k]

        ax.set_ylim(0.95 * min([min(x) for x in y if len(x) > 0]),
                    1.1 * max([max(x) for x in y if len(x) > 0]))

    plt.title("Flare in " + str(int(search_window)) + " day window")

    ax1.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.tight_layout()
    plt.savefig(plot_output_dir(name) + "/flare_vs_box_" +
                catalogue["source_name"][0].decode() + "_" + ["sens", "disc"][
                    j] +
                ".pdf")
    plt.close()

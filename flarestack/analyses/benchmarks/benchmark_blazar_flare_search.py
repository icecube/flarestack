"""
Script to reproduce the analysis of the 1ES 1959+650 blazar, as described in
https://wiki.icecube.wisc.edu/index.php/1ES_1959_Analysis.

The script can be used to verify that the flare search method, as implemented
here, is capable of matching previous flare search methods.
"""

import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import gfu_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    transients_dir
from flarestack.utils.prepare_catalogue import custom_sources
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from astropy.coordinates import Distance

name = "analyses/benchmarks/1ES_blazar_benchmark/"

analyses = dict()

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

# Start and end time of flare in MJD
t_start = 57506.00
t_end = 57595.00

# Ra and dec of source
ra = 300.00
dec = 65.15

# Distance to source -> z=0.048 according to TeVCat. With lambdaCDM, this gives:
z = 0.048
lumdist = Distance(z=z).to("Mpc").value

# Creates the .npy source catalogue
catalogue = custom_sources(
    name="1ES_1959+650",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=lumdist,
    start_time=t_start,
    end_time=t_end,
)

cat_path = transients_dir + "1ES_1959+650.npy"
np.save(cat_path, catalogue)

max_window = float(t_end - t_start)

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": t_start,
    "Pre-Window": 0.,
    "Post-Window": max_window,
    "Max Flare": 21.
}

llh_energy = injection_energy

no_flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": False
}

no_flare_negative = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": True
}

flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True,
    "Fit Negative n_s?": False
}

src_res = dict()

lengths = np.logspace(-2, 0, 5) * max_window

for i, llh_kwargs in enumerate([
                                no_flare,
                                no_flare_negative,
                                flare
                                ]):

    label = ["Time-Independent", "Time-Independent (negative n_s)",
             "Time-Clustering"][i]
    f_name = ["fixed_box", "fixed_box_negative", "flare_fit_gamma"][i]

    flare_name = name + f_name + "/"

    res = dict()

    for flare_length in lengths:

        full_name = flare_name + str(flare_length) + "/"

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

        scale = flux_to_k(reference_sensitivity(np.sin(dec))
                          * (70 * max_window / flare_length))

        mh_dict = {
            "name": full_name,
            "datasets": gfu_v002_p01,
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

        rd.submit_to_cluster(pkl_file, n_jobs=5000)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=1)
        # mh.clear()

        # raw_input("prompt")

        res[flare_length] = mh_dict

    src_res[label] = res

rd.wait_for_cluster()

sens = [[] for _ in src_res]
fracs = [[] for _ in src_res]
disc_pots = [[] for _ in src_res]
sens_e = [[] for _ in src_res]
disc_e = [[] for _ in src_res]

labels = []

for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):
    for (length, rh_dict) in sorted(res.iteritems()):

        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                            rh_dict["catalogue"])

        inj_time = length * (60 * 60 * 24)

        astro_sens, astro_disc = rh.astro_values(
            rh_dict["inj kwargs"]["Injection Energy PDF"])

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

        print fluence[i], labels[i]

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

    plt.title("Flare in " + str(int(max_window)) + " day window")

    ax1.legend(loc='upper left', fancybox=True, framealpha=0.)
    plt.tight_layout()
    plt.savefig(plot_output_dir(name) + "/flare_vs_box_" +
                catalogue["Name"][0] + "_" + ["sens", "disc"][j] + ".pdf")
    plt.close()

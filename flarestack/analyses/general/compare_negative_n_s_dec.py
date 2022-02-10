from __future__ import division
from builtins import str
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 100.0

llh_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": 56100.0,
    "Pre-Window": 0.0,
    "Post-Window": max_window,
}

llh_energy = injection_energy

zero_bound = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": False,
}

negative_bound = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": True,
}

name = "benchmarks/negative_n_s/"

sindecs = np.linspace(0.90, -0.90, 13)
# sindecs = np.linspace(0.5, -0.5, 3)
# sindecs = [0.0]

length = 100.0

analyses = dict()

for i, llh_kwargs in enumerate([zero_bound, negative_bound]):

    label = ["Zero-bound", "Negative-bound"][i]

    src_res = dict()

    for sindec in sindecs:

        cat_path = ps_catalogue_name(sindec)

        decname = name + "sindec=" + "{0:.2f}".format(sindec) + "/"

        full_name = decname + label + "/"

        injection_time = dict(llh_time)
        injection_time["Post-Window"] = length

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
            "Poisson Smear?": True,
        }

        scale = flux_to_k(reference_sensitivity(sindec)) * (40)

        mh_dict = {
            "name": full_name,
            "datasets": [ps_v002_p01[-1]],
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 10,
            "n_steps": 10,
        }

        analysis_path = analysis_dir + full_name

        try:
            os.makedirs(analysis_path)
        except OSError:
            pass

        pkl_file = analysis_path + "dict.pkl"

        with open(pkl_file, "wb") as f:
            Pickle.dump(mh_dict, f)
        #
        rd.submit_to_cluster(pkl_file, n_jobs=500)

        # if label == "Negative-bound":
        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], n_steps=3, n_trials=100)

        src_res[sindec] = mh_dict

    analyses[label] = src_res

rd.wait_for_cluster()

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

plt.title("Sensitivity for " + str(int(max_window)) + " day window")

all_sens = []
all_fracs = []


for i, (label, src_res) in enumerate(analyses.items()):

    cols = ["b", "orange", "g"]

    sens = []
    fracs = []
    disc_pots = []

    for (sindec, rh_dict) in sorted(src_res.items()):

        try:
            #
            rh = ResultsHandler(rh_dict)
            sens.append(rh.sensitivity)
            disc_pots.append(rh.disc_potential)
        except EOFError:
            sens.append(np.nan)
            disc_pots.append(np.nan)

        fracs.append(sindec)

    mask = ~np.isnan(sens)

    fracs = np.array(fracs)
    sens = np.array(sens)
    disc_pots = np.array(disc_pots)

    ax1.plot(fracs[mask], sens[mask], label=label, color=cols[i])
    ax1.plot(fracs[mask], disc_pots[mask], linestyle="--", color=cols[i])

    all_sens.append(sens)

ax1.grid(True, which="both")
ax1.set_yscale("log")
ax1.set_xlim(xmin=-1.0, xmax=1.0)
yticks = ax1.yaxis.get_major_ticks()
yticks[1].label1.set_visible(False)

ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]", fontsize=12)

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

ratios = np.array(all_sens[1]) / np.array(all_sens[0])

mask = ~np.isnan(ratios)

ax2.scatter(sindecs[mask], ratios[mask], color="red")
ax2.plot(sindecs[mask], ratios[mask], color="red")
ax2.set_ylabel(r"Ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)
# ax2.set_ylim(ymin=0.5, ymax=1.5)
ax2.grid(True)
xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

# ax1.set_xscale("log")
# ax1.set_xlim(0, 1.0)

ax1.legend(loc="upper right", fancybox=True, framealpha=1.0)

savename = plot_output_dir(name) + "ratio.pdf"

try:
    os.makedirs(os.path.dirname(savename))
except OSError:
    pass


plt.savefig(savename)
plt.close()

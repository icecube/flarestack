import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 100.

llh_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": 56100.,
    "Pre-Window": 0.,
    "Post-Window": max_window
}

llh_energy = injection_energy

zero_bound = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Brute Seed?": False
}

negative_bound = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Brute Seed?": True
}

name = "benchmarks/brute_seed/"

sindecs = np.linspace(0.90, -0.90, 13)
# sindecs = np.linspace(0.5, -0.5, 2)
# sindecs = [0.0]

length = 100.


analyses = dict()

for i, llh_kwargs in enumerate([zero_bound, negative_bound]):
    label = ["Default-seed", "Brute-seed"][i]

    src_res = dict()

    for sindec in sindecs:

        cat_path = ps_catalogue_name(sindec)

        decname = name + "sindec=" + '{0:.2f}'.format(sindec) + "/"

        full_name = decname + label + "/"

        injection_time = dict(llh_time)
        injection_time["Post-Window"] = length

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
            "Poisson Smear?": True,
        }

        scale = flux_to_k(reference_sensitivity(sindec)) * 40

        mh_dict = {
            "name": full_name,
            "datasets": [ps_7year[-1]],
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 5,
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

        # rd.submit_to_cluster(pkl_file, n_jobs=2000)

        # mh = MinimisationHandler(mh_dict)

        # if label == "Negative-bound":
        #     for j in np.linspace(0., scale, 3):
        #         mh.scan_likelihood(j)
        # if i > 0:
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=10)

        src_res[sindec] = mh_dict

    analyses[label] = src_res

rd.wait_for_cluster()

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

plt.title("Sensitivity for " + str(int(max_window)) + " day window")

all_sens = []

for i, (label, src_res) in enumerate(analyses.iteritems()):

    cols = ["b", "orange", "g"]

    sens = []
    fracs = []
    disc_pots = []

    for (sindec, rh_dict) in sorted(src_res.iteritems()):
        #
        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                            rh_dict["catalogue"])
        sens.append(rh.sensitivity)
        disc_pots.append(rh.disc_potential)
        fracs.append(sindec)

    ax1.plot(fracs, sens, label=label, color=cols[i])

    all_sens.append(sens)

ax1.grid(True, which='both')
ax1.set_yscale("log")
ax1.set_xlim(xmin=-1., xmax=1.)
yticks = ax1.yaxis.get_major_ticks()
yticks[1].label1.set_visible(False)

ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
               fontsize=12)

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

ratios = np.array(all_sens[1]) / np.array(all_sens[0])

ax2.scatter(sindecs, ratios, color="black")
ax2.plot(sindecs, ratios, linestyle="--", color="red")
ax2.set_ylabel(r"ratio", fontsize=12)
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

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

savename = plot_output_dir(name) + "ratio.pdf"

try:
    os.makedirs(os.path.dirname(savename))
except OSError:
    pass


plt.savefig(savename)
plt.close()

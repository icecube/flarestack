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
from flarestack.utils.custom_seasons import custom_dataset

name_root = "benchmarks/timePDFs/scaling/"

# Picks ref time at start of IC86-I

ref_time = 55700
window = 100

analyses = dict()

# Shared

energy_pdf = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

# Initialise Injection

inj_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": ref_time,
    "Pre-Window": 0,
    "Post-Window": window
}

inj = {
    "Injection Time PDF": inj_time,
    "Injection Energy PDF": energy_pdf,
    "Poisson Smear?": True
}

cat_res = dict()

sindecs = np.linspace(0.5, -0.5, 3)
# sindecs = [0.0]

lengths = np.logspace(-1, 1, 11) * window

print lengths
raw_input("prompt")

for sindec in sindecs:

    source_res = dict()

    cat_path = ps_catalogue_name(sindec)

    sindec_key = "sindec=" + '{0:.2f}'.format(sindec)

    name = name_root + sindec_key + "/"

    src_res = dict()

    for length in lengths:

        full_name = name + str(length) + "/"

        scale = flux_to_k(reference_sensitivity(sindec) * 20) * max(
            1, window / abs(length)
        )

        # Standard Time Integration

        llh_time = {
            "Name": "FixedRefBox",
            "Fixed Ref Time (MJD)": ref_time,
            "Pre-Window": 0,
            "Post-Window": length
        }

        llh_kwargs = {
            "LLH Energy PDF": energy_pdf,
            "LLH Time PDF": llh_time,
            "Fit Gamma?": True,
            "Fit Negative n_s?": True,
            "Fit Weights?": False
        }

        mh_dict = {
            "name": full_name,
            "datasets": custom_dataset(ps_7year, np.load(cat_path),
                                       llh_kwargs["LLH Time PDF"]),
            "catalogue": cat_path,
            "inj kwargs": inj,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 100,
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

        rd.submit_to_cluster(pkl_file, n_jobs=20)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=20)
        # mh.clear()

        src_res[length] = mh_dict

    cat_res[sindec_key] = src_res

rd.wait_for_cluster()

plt.figure()
ax = plt.subplot(111)

cols = ["r", "g", "b", "orange"]

for i, (sindec_key, src_res) in enumerate(cat_res.iteritems()):

    name = name_root + sindec_key + "/"

    sens = []
    lengths = []

    for (t, rh_dict) in sorted(src_res.iteritems()):
        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                            rh_dict["catalogue"])
        sens.append(rh.sensitivity)
        lengths.append(t)

    ax.plot(lengths, sens, label=sindec_key, color=cols[i])

ax.set_ylabel(r"Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=12)
ax.set_xlabel("Time-Integration Window (days)")
ax.set_yscale("log")
# plt.legend()

plt.title("Sensitivity for 100 day emission")

ax.legend(loc='upper right', fancybox=True, framealpha=1.)
plt.tight_layout()
plt.savefig(plot_output_dir(name_root) + "scaling_sens.pdf")
plt.close()

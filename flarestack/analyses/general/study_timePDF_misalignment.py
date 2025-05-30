from __future__ import division

import os
import pickle as Pickle
from builtins import str

import matplotlib.pyplot as plt
import numpy as np

from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.shared import analysis_dir, flux_to_k, plot_output_dir
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.utils.prepare_catalogue import ps_catalogue_name

name_root = "benchmarks/timePDFs/misalignment/"

# Picks ref time ~100 days into IC86-I

ref_time = 55800
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
    "Post-Window": window,
}

inj = {
    "Injection Time PDF": inj_time,
    "Injection Energy PDF": energy_pdf,
    "Poisson Smear?": True,
}

cat_res = dict()

sindecs = np.linspace(0.5, -0.5, 3)
# sindecs = [0.0]

offsets = np.linspace(-90.0, 90, 7)

for sindec in sindecs:
    source_res = dict()

    cat_path = ps_catalogue_name(sindec)

    sindec_key = "sindec=" + "{0:.2f}".format(sindec)

    name = name_root + sindec_key + "/"

    src_res = dict()

    for offset in offsets:
        full_name = name + str(offset) + "/"

        scale = flux_to_k(reference_sensitivity(sindec) * 20) * (
            window / (window - abs(offset))
        )

        # Standard Time Integration

        llh_time = {
            "Name": "FixedRefBox",
            "Fixed Ref Time (MJD)": 55800 + offset,
            "Pre-Window": 0,
            "Post-Window": window,
        }

        llh_kwargs = {
            "LLH Energy PDF": energy_pdf,
            "LLH Time PDF": llh_time,
            "Fit Gamma?": True,
            "Fit Negative n_s?": True,
            "Fit Weights?": False,
        }

        mh_dict = {
            "name": full_name,
            "datasets": custom_dataset(
                ps_v002_p01, np.load(cat_path), llh_kwargs["LLH Time PDF"]
            ),
            "catalogue": cat_path,
            "inj kwargs": inj,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 100,
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

        # rd.submit_to_cluster(pkl_file, n_jobs=20)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=100)
        # mh.clear()

        src_res[offset] = mh_dict

    cat_res[sindec_key] = src_res

# rd.wait_for_cluster()

plt.figure()
ax = plt.subplot(111)

cols = ["r", "g", "b", "orange"]

for i, (sindec_key, src_res) in enumerate(cat_res.items()):
    name = name_root + sindec_key + "/"

    sens = []
    offsets = []

    for t, rh_dict in sorted(src_res.items()):
        rh = ResultsHandler(rh_dict)
        sens.append(rh.sensitivity)
        offsets.append(t)

    ax.plot(offsets, sens, label=sindec_key, color=cols[i])

ax.set_ylabel(r"Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=12)
ax.set_xlabel("Offset (days)")
ax.set_yscale("log")
# plt.legend()

plt.title("Sensitivity for 100 day emission")

ax.legend(loc="upper left", fancybox=True, framealpha=1.0)
plt.tight_layout()
plt.savefig(plot_output_dir(name_root) + "misalignment_sens.pdf")
plt.close()

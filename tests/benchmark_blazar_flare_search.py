"""
Script to reproduce the analysis of the 1ES 1959+650 blazar, as described in x.
The script can be used to verify that the flare search method, as implemented
here, is capable of matching previous flare search methods.
"""

import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_gfu_2point5_year import gfu_2point5, ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.prepare_catalogue import custom_sources, ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from scipy.interpolate import interp1d
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

name = "tests/1ES_blazar_benchmark/"

analyses = dict()

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

catalogue = custom_sources(
    ra=300.00,
    dec=65.15,
    weight=1.,
    distance=1.,
    ref_time=57507.00,
    start_time=57507.00,
    end_time=57595.00,
    name="1ES_1959+650"
)

cat_path = ps_catalogue_name(0.0)

source = np.load(cat_path)

t_s = source["Start Time (MJD)"]
t_e = source["End Time (MJD)"]

max_window = float(t_e - t_s)

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 100.

llh_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": 57507.00,
    "Pre-Window": 0.,
    "Post-Window": max_window
}
llh_energy = injection_energy

no_flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Find Flare?": False
}

flare_with_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True
}

src_res = dict()

lengths = np.linspace(0.0, 1.0, 3)[1:] * max_window

# lengths = [0.5 * max_window]

for i, llh_kwargs in enumerate([no_flare]):

    label = ["Time-Integrated", "Flare"][i]
    f_name = ["fixed_box", "flare_fit_gamma"][i]

    flare_name = name + f_name + "/"

    res = dict()

    for flare_length in lengths:

        full_name = flare_name + str(flare_length) + "/"

        injection_time = dict(llh_time)
        injection_time["Post-Window"] = flare_length

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
            "Poisson Smear?": True,
        }

        scale = flux_to_k(skylab_7year_sensitivity(np.sin(source["dec"]))
                          * (40 * max_window / flare_length))

        mh_dict = {
            "name": full_name,
            "datasets": gfu_2point5,
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 3,
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

        # rd.submit_to_cluster(pkl_file, n_jobs=2000)

        mh = MinimisationHandler(mh_dict)
        mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=500)

        del mh

        res[flare_length] = mh_dict

    src_res[label] = res

rd.wait_for_cluster()

plt.figure()
ax1 = plt.subplot(111)

cols = ["r", "g", "b"]

for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):

    if f_type != "Flare (fixed Gamma)":
        sens = []
        fracs = []
        disc_pots = []

        for (length, rh_dict) in sorted(res.iteritems()):
            try:
                rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                    rh_dict["catalogue"])
                sens.append(rh.sensitivity * float(length) * 60 * 60 * 24)
                disc_pots.append(rh.disc_potential *
                                 float(length) * 60 * 60 * 24)
                fracs.append(length)

            except OSError:
                pass

        plt.plot(fracs, sens, label=f_type, color=cols[i])
    # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

ax1.grid(True, which='both')
# ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Fluence [ GeV$^{-1}$ cm$^{-2}$]",
               fontsize=12)
ax1.set_xlabel(r"Flare Length (days)")
# ax1.set_xscale("log")
# ax1.set_xlim(0, 1.0)

plt.title("Flare in " + str(int(max_window)) + " day window")

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
plt.savefig(plot_output_dir(name) + "/flare_vs_box.pdf")
plt.close()

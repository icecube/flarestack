import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7_year import IC86_234_dict
from shared import plot_output_dir, flux_to_k, analysis_dir
from utils.prepare_catalogue import ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from scipy.interpolate import interp1d
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 300.

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
    "Fit Negative n_s?": False
}

negative_bound = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": True
}

name = "tests/negative_n_s/"

# sindecs = np.linspace(0.90, -0.90, 13)
sindecs = np.linspace(0.5, -0.5, 3)
sindecs = [0.0]

length = 100.


analyses = dict()

for i, llh_kwargs in enumerate([zero_bound, negative_bound]):

    label = ["Zero-bound", "Negative-bound"][i]

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

        scale = flux_to_k(skylab_7year_sensitivity(sindec)) * (40)

        mh_dict = {
            "name": full_name,
            "datasets": [IC86_234_dict],
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
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=3)

        src_res[sindec] = mh_dict

    analyses[label] = src_res

# rd.wait_for_cluster()

plt.figure()
ax1 = plt.subplot(111)

for i, (label, src_res) in enumerate(analyses.iteritems()):

    cols = ["r", "g", "b"]

    sens = []
    fracs = []
    disc_pots = []

    for (sindec, rh_dict) in sorted(analyses.iteritems()):

        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                            rh_dict["catalogue"])
        sens.append(rh.sensitivity * float(length) * 60 * 60 * 24)
        disc_pots.append(rh.disc_potential *
                         float(length) * 60 * 60 * 24)
        fracs.append(float(length)/max_window)

    plt.plot(fracs, sens, label=label, color=cols[i])


ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Fluence [ GeV$^{-1}$ cm$^{-2}$]",
               fontsize=12)
ax1.set_xlabel(r"(Flare Length) / (Maximum Window)")
# ax1.set_xscale("log")
# ax1.set_xlim(0, 1.0)

plt.title("Flare in " + str(int(max_window)) + " day window")

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

savename = plot_output_dir(name) + "ratio.pdf"

try:
    os.makedirs(os.path.dirname(savename))
except OSError:
    pass


plt.savefig(savename)
plt.close()

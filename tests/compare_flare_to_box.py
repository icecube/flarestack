import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7year import IC86_234_dict
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

no_flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Find Flare?": True
}

flare_no_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": False,
    "Find Flare?": True
}

flare_with_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Find Flare?": True
}

name = "tests/flare_vs_window"

# sindecs = np.linspace(0.90, -0.90, 13)
sindecs = np.linspace(0.5, -0.5, 3)
sindecs = [0.0]

lengths = np.linspace(0, max_window, 31)

print lengths
raw_input("prompt")
# lengths = [20., 50.]



analyses = dict()

for sindec in sindecs:

    cat_path = ps_catalogue_name(sindec)

    decname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    src_res = dict()

    for i, llh_kwargs in enumerate([no_flare, flare_no_energy,
                                    flare_with_energy]):

        label = ["Fixed Box", "Flare (fixed Gamma)", "Flare (fit Gamma)"][i]
        f_name = ["fixed_box", "flare_fixed_gamma", "flare_fit_gamma"][i]

        flare_name = decname + f_name + "/"

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

            scale = flux_to_k(skylab_7year_sensitivity(sindec)) * (
                    4000 / flare_length)

            mh_dict = {
                "name": full_name,
                "datasets": [IC86_234_dict],
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
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

            # rd.submit_to_cluster(pkl_file, n_jobs=100)

            mh = MinimisationHandler(mh_dict)
            mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=1)

            res[flare_length] = mh_dict

        src_res[label] = res

    analyses[sindec] = src_res

rd.wait_for_cluster()

for (sindec, src_res) in analyses.iteritems():
    plt.figure()
    ax1 = plt.subplot(111)

    for (f_type, res) in sorted(src_res.iteritems()):
        sens = []
        fracs = []

        for (length, rh_dict) in sorted(res.iteritems()):
            try:
                rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                    rh_dict["catalogue"], cleanup=True)
                sens.append(rh.sensitivity * float(length) * 60 * 60 * 24)
                fracs.append(float(length)/max_window)
            except:
                pass

        plt.plot(fracs, sens, label=f_type)

    ax1.grid(True, which='both')
    ax1.semilogy(nonposy='clip')
    ax1.set_ylabel(r"Time-Integrated Flux Strength [ GeV$^{-1}$ cm$^{-2}$]",
                   fontsize=12)
    ax1.set_xlabel(r"(Flare Length) / (Maximum Window)")
    ax1.set_xlim(0, 1.0)

    plt.title("Flare in " + str(int(max_window)) + " day window")

    ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
    plt.savefig(plot_output_dir(name) + "/sindec=" + '{0:.2f}'.format(sindec)
                + "/flare_vs_box.pdf")
    plt.close()

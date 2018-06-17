"""
Script to reproduce the analysis of the 1ES 1959+650 blazar, as described in
https://wiki.icecube.wisc.edu/index.php/1ES_1959_Analysis.

The script can be used to verify that the flare search method, as implemented
here, is capable of matching previous flare search methods.
"""

import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_diffuse_8year import diffuse_8year
from data.icecube_gfu_2point5_year import gfu_2point5
from shared import plot_output_dir, flux_to_k, analysis_dir, transients_dir
from utils.prepare_catalogue import custom_sources, ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF

name = "tests/1ES_blazar_benchmark/"

analyses = dict()

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

# Start and end time of flare in MJD
t_start = 57506.00
t_end = 57595.00

# Ra and dec of source
ra = 300.00
dec = 65.15

# Creates the .npy source catalogue
catalogue = custom_sources(
    name="1ES_1959+650",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=1.,
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
    "Post-Window": max_window
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

flare_with_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True,
    "Fit Negative n_s?": False
}

src_res = dict()

lengths = np.array(sorted([0.05] + list(np.linspace(0.0, 1.0, 11)))[1:]) * \
                 max_window

# lengths = [0.5 * max_window]

for i, llh_kwargs in enumerate([no_flare, no_flare_negative,
                                flare_with_energy]):

    label = ["Time-Integrated", "Time-Integrated (negative n_s)", "Flare"][i]
    f_name = ["fixed_box", "fixed_box_negative", "flare_fit_gamma"][i]

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

        scale = flux_to_k(skylab_7year_sensitivity(np.sin(dec))
                          * (50 * max_window / flare_length))

        mh_dict = {
            "name": full_name,
            "datasets": gfu_2point5,
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

        injection_time = mh_dict["inj kwargs"]["Injection Time PDF"]

        inj_time = 0.

        for season in mh_dict["datasets"]:
            time = TimePDF.create(injection_time, season)
            inj_time += time.effective_injection_time(catalogue)

        print "Injecting for", flare_length, "Livetime", inj_time/(60.*60.*24.)

        # rd.submit_to_cluster(pkl_file, n_jobs=5000)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=1)
        # mh.clear()
        res[flare_length] = mh_dict

    src_res[label] = res

# rd.wait_for_cluster()

sens = [[] for _ in src_res]
sens_livetime = [[] for _ in src_res]
fracs = [[] for _ in src_res]
disc_pots = [[] for _ in src_res]
disc_pots_livetime = [[] for _ in src_res]

labels = []

for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):
    for (length, rh_dict) in sorted(res.iteritems()):
        try:
            rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                rh_dict["catalogue"])

            # The uptime noticeably deviates from 100%, because the detector
            # was undergoing tests for 25 hours on May 5th/6th 2016. Thus,
            # particularly for short flares, the sensitivity appears to
            # improve as a function of time unless this is taken into account.
            injection_time = rh_dict["inj kwargs"]["Injection Time PDF"]

            inj_time = 0.

            for season in rh_dict["datasets"]:
                time = TimePDF.create(injection_time, season)
                inj_time += time.effective_injection_time(catalogue)

            sens[i].append(rh.sensitivity * float(length) * 60 * 60 * 24)
            disc_pots[i].append(rh.disc_potential *
                                float(length) * 60 * 60 * 24)
            sens_livetime[i].append(rh.sensitivity * inj_time)
            disc_pots_livetime[i].append(rh.disc_potential * inj_time)
            fracs[i].append(length)

        except OSError:
            pass

        except KeyError:
            pass

        except EOFError:
            pass

    labels.append(f_type)
    # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

for j, s in enumerate([sens, sens_livetime]):

    d = [disc_pots, disc_pots_livetime][j]

    for k, y in enumerate([s, d]):

        plt.figure()
        ax1 = plt.subplot(111)

        cols = ["r", "g", "b"]
        linestyle = ["-", "--"][k]

        for i, f in enumerate(fracs):
            plt.plot(f, y[i], label=labels[i], linestyle=linestyle,
                     color=cols[i])

        label = ["", "(Livetime-adjusted)"][j]

        ax1.grid(True, which='both')
        # ax1.semilogy(nonposy='clip')
        ax1.set_ylabel(r"Fluence [ GeV$^{-1}$ cm$^{-2}$]", fontsize=12)
        ax1.set_xlabel(r"Flare Length (days)")
        # ax1.set_xscale("log")
        ax1.set_ylim(0.95 * min([min(x) for x in y]),
                     1.1 * max([max(x) for x in y]))

        plt.title("Flare in " + str(int(max_window)) + " day window")

        ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
        plt.savefig(plot_output_dir(name) + "/flare_vs_box" + label + "_" +
                    ["sens", "disc"][k] + ".pdf")
        plt.close()

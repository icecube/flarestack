import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7_year import ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.skylab_reference import skylab_7year_sensitivity
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF

name = "analyses/tde/compare_fitting_weights/"

analyses = dict()

# Start and end time of flare in MJD
t_start = 55740.00
t_end = 55840.00

cat_path = catalogue_dir + "TDEs/Jetted_TDE_catalogue.npy"
# cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
# cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
catalogue = np.load(cat_path)

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

# llh_time = {
#     "Name": "Steady"
# }

llh_energy = injection_energy

# fit_weights_neg = {
#     "LLH Energy PDF": llh_energy,
#     "LLH Time PDF": llh_time,
#     "Fit Gamma?": True,
#     "Fit Negative n_s?": True,
#     "Fit Weights?": True
# }
#
# fixed_weights_neg = {
#     "LLH Energy PDF": llh_energy,
#     "LLH Time PDF": llh_time,
#     "Fit Gamma?": True,
#     "Fit Negative n_s?": True,
#     "Fit Weights?": False
# }

fit_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
    # "Fit Weights?": True
}

fixed_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
    "Fit Weights?": False
}


src_res = dict()

lengths = np.array(sorted([0.05] + list(np.linspace(0.0, 1.0, 3)))[1:]) * \
                 max_window

# lengths = [0.5 * max_window]

for i, llh_kwargs in enumerate([fit_weights, fixed_weights]):
    label = ["Fit Weights", "Fixed Weights"][i]
    f_name = ["fit_weights", "fixed_weights"][i]

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

        scale = flux_to_k(skylab_7year_sensitivity(np.sin(0.))
                          * (25 * max_window / flare_length))

        print scale

        mh_dict = {
            "name": full_name,
            "datasets": ps_7year[-2:-1],
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 1,
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

        injection_time = mh_dict["inj kwargs"]["Injection Time PDF"]

        inj_time = 0.

        for season in mh_dict["datasets"]:
            time = TimePDF.create(injection_time, season)
            inj_time += time.effective_injection_time(catalogue)

        print "Injecting for", flare_length, "Livetime", inj_time/(60.*60.*24.)

        # rd.submit_to_cluster(pkl_file, n_jobs=2000)
        #
        mh = MinimisationHandler(mh_dict)
        mh.scan_likelihood(scale=100)
        raw_input("done")
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=10)
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
            # disc_pots[i].append(rh.disc_potential *
            #                     float(length) * 60 * 60 * 24)
            sens_livetime[i].append(rh.sensitivity * inj_time)
            # disc_pots_livetime[i].append(rh.disc_potential * inj_time)
            fracs[i].append(length)

        except OSError:
            pass

    labels.append(f_type)
    # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

for j, s in enumerate([sens, sens_livetime]):

    d = [disc_pots, disc_pots_livetime][j]

    plt.figure()
    ax1 = plt.subplot(111)

    cols = ["r", "g", "b"]

    for i, f in enumerate(fracs):
        plt.plot(f, s[i], label=labels[i], color=cols[i])
        # plt.plot(f, d[i], linestyle="--", color=cols[i])

    label = ["", "(Livetime-adjusted)"][j]

    ax1.grid(True, which='both')
    ax1.semilogy(nonposy='clip')
    ax1.set_ylabel(r"Fluence [ GeV$^{-1}$ cm$^{-2}$] " + label,
                   fontsize=12)
    ax1.set_xlabel(r"Flare Length (days)")
    # ax1.set_xscale("log")
    # ax1.set_ylim(0.95 * min([min(x) for x in s]),
    #              1.1 * max([max(x) for x in s]))

    plt.title("Flare in " + str(int(max_window)) + " day window")

    ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
    plt.savefig(plot_output_dir(name) + "/flare_vs_box" + label + ".pdf")
    plt.close()

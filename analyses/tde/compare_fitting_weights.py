import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_ps_tracks_v002_p01 import ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.reference_sensitivity import reference_sensitivity
from cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF

analyses = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox",
}

llh_energy = injection_energy

fit_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Weights?": True
}

fixed_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Fit Weights?": False
}

fixed_weights_negative = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": True,
    "Fit Weights?": False
}

flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True,
    "Fit Negative n_s?": False
}

cat_res = dict()

# Inject for, at most, 100 days

max_window = 100

lengths = np.logspace(-2, 0, 5) * max_window

for cat in ["jetted"]:

    name = "analyses/tde/compare_fitting_weights/" + cat + "/"

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_res = dict()

    # lengths = [0.5 * max_window]

    for i, llh_kwargs in enumerate([
                                    fixed_weights,
                                    fixed_weights_negative,
                                    fit_weights,
                                    # flare
                                    ]):
        label = ["Fixed Weights", "Fixed Weights (Negative n_s)",
                 "Fit Weights", "Flare Search", ][i]
        f_name = ["fixed_weights", "fixed_weights_neg",
                  "fit_weights", "flare"][i]

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

            scale = 60 * max_window / flare_length

            # print scale

            mh_dict = {
                "name": full_name,
                "datasets": ps_7year[-3:-1],
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 5,
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

            # rd.submit_to_cluster(pkl_file, n_jobs=100)
            # #
            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=1)
            # mh.clear()
            # raw_input("prompt")

            res[flare_length] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

rd.wait_for_cluster()

for (cat, src_res) in cat_res.iteritems():

    name = "analyses/tde/compare_fitting_weights/" + cat + "/"

    sens = [[] for _ in src_res]
    sens_livetime = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    disc_pots_livetime = [[] for _ in src_res]

    labels = []

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_1_frac = min(catalogue["Distance (Mpc)"])**-2/np.sum(
        catalogue["Distance (Mpc)"] ** -2
    )

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

                fracs[i].append(length)

            except OSError:
                pass

        labels.append(f_type)
        # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    for j, s in enumerate([sens, sens_livetime]):

        d = [disc_pots, disc_pots_livetime][j]

        for k, y in enumerate([s, d]):

            plt.figure()
            ax1 = plt.subplot(111)

            cols = ["orange", "g", "b", "r"]
            linestyle = ["-", "--"][k]

            for i, f in enumerate(fracs):
                plt.plot(f, y[i], label=labels[i], linestyle=linestyle,
                         color=cols[i])

            label = ["", "(Livetime-adjusted)"][j]

            ax1.grid(True, which='both')
            # ax1.semilogy(nonposy='clip')
            ax1.set_ylabel(r"Time-Integrated Flux [ GeV$^{-1}$ cm$^{-2}$]",
                           fontsize=12)
            ax1.set_xlabel(r"Flare Length (days)")
            # ax1.set_xscale("log")
            ax1.set_ylim(0.95 * min([min(x) for x in y]),
                         1.1 * max([max(x) for x in y]))

            plt.title("Flare in " + str(int(max_window)) + " day window")

            ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
            plt.savefig(plot_output_dir(name) + "/flare_vs_box" + label + "_" +
                        ["sens", "disc"][k] + ".pdf")
            plt.close()
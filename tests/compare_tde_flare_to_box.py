import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7_year import ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.prepare_catalogue import ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from scipy.interpolate import interp1d
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF


TDEs = ["Swift J1644+57"]

name = "tests/flare_vs_window/"

# print ps_7year


analyses = dict()

for tde in TDEs:

    cat_path = catalogue_dir + "/TDEs/individual_TDEs/" + tde + "_catalogue.npy"

    source = np.load(cat_path)

    t_s = source["Start Time (MJD)"]
    t_e = source["End Time (MJD)"]

    max_window = float(t_e - t_s)

    # Initialise Injectors/LLHs

    injection_energy = {
        "Name": "Power Law",
        "Gamma": 2.0,
    }

    # max_window = 100.

    llh_time = {
        "Name": "FixedRefBox",
        "Fixed Ref Time (MJD)": t_s,
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

    flare_no_energy = {
        "LLH Energy PDF": llh_energy,
        "LLH Time PDF": llh_time,
        "Fit Gamma?": False,
        "Flare Search?": True
    }

    flare_with_energy = {
        "LLH Energy PDF": llh_energy,
        "LLH Time PDF": llh_time,
        "Fit Gamma?": True,
        "Flare Search?": True
    }

    tdename = name + tde + "/"

    src_res = dict()

    lengths = np.linspace(0.0, 1.0, 11)[1:] * max_window

    # lengths = [0.5 * max_window]

    for i, llh_kwargs in enumerate([no_flare, flare_no_energy,
                                    flare_with_energy]):

        label = ["Time-Integrated", "Flare (fixed Gamma)", "Flare"][i]
        f_name = ["fixed_box", "flare_fixed_gamma", "flare_fit_gamma"][i]

        flare_name = tdename.replace(" ", "") + f_name + "/"

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
                "datasets": ps_7year,
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

            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=1)

            res[flare_length] = mh_dict

            inj_time = 0.

            for season in mh_dict["datasets"]:
                time = TimePDF.create(injection_time, season)
                inj_time += time.effective_injection_time(source)

            print "Injecting for", flare_length, "Livetime", inj_time / (
                        60. * 60. * 24.)

        if label != "Flare (fixed Gamma)":
            src_res[label] = res

    analyses[tde] = src_res

# rd.wait_for_cluster()

for (tde, src_res) in analyses.iteritems():

    sens = [[] for _ in src_res]
    sens_livetime = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    disc_pots_livetime = [[] for _ in src_res]

    labels = []

    for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):

        if f_type != "Flare (fixed Gamma)":

            for (length, rh_dict) in sorted(res.iteritems()):
                try:
                    rh = ResultsHandler(rh_dict["name"],
                                        rh_dict["llh kwargs"],
                                        rh_dict["catalogue"])

                    catalogue = np.load(rh_dict["catalogue"])

                    # The uptime noticeably deviates from 100%, because the detector
                    # was undergoing tests for 25 hours on May 5th/6th 2016. Thus,
                    # particularly for short flares, the sensitivity appears to
                    # improve as a function of time unless this is taken into account.
                    injection_time = rh_dict["inj kwargs"]["Injection Time PDF"]

                    inj_time = 0.

                    for season in rh_dict["datasets"]:
                        time = TimePDF.create(injection_time, season)
                        inj_time += time.effective_injection_time(catalogue)

                    sens[i].append(
                        rh.sensitivity * float(length) * 60 * 60 * 24)
                    disc_pots[i].append(rh.disc_potential *
                                        float(length) * 60 * 60 * 24)
                    sens_livetime[i].append(rh.sensitivity * inj_time)
                    disc_pots_livetime[i].append(
                        rh.disc_potential * inj_time)
                    fracs[i].append(length)

                except OSError:
                    pass

        labels.append(f_type)

            # plt.plot(fracs, sens, label=f_type, color=cols[i])
            # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    for j, s in enumerate([sens, sens_livetime]):

        d = [disc_pots, disc_pots_livetime][j]

        for k, y in enumerate([s, d]):

            plt.figure()
            ax1 = plt.subplot(111)

            cols = ["r", "g", "b"]
            linestyle = ["-", "--"][k]

            max_window = max(max(fracs))

            for l, f in enumerate(fracs):
                plt.plot(f, y[l], label=labels[l], linestyle=linestyle,
                         color=cols[l])

            label = ["", "(Livetime-adjusted)"][j]

            ax1.grid(True, which='both')
            # ax1.semilogy(nonposy='clip')
            ax1.set_ylabel(r"Fluence [ GeV$^{-1}$ cm$^{-2}$]", fontsize=12)
            ax1.set_xlabel(r"Flare Length (days)")
            # ax1.set_xscale("log")

            print y

            ax1.set_ylim(0.95 * min([min(x) for x in np.array(y)]),
                         1.1 * max([max(x) for x in np.array(y)]))

            plt.title("Flare in " + str(max_window) + " day window")

            ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
            plt.savefig(plot_output_dir(name) + "/" + tde.replace(" ", "") +
                        "/flare_vs_box" + label + "_" +
                        ["sens", "disc"][k] + ".pdf")
            plt.close()

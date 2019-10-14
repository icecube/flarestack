from __future__ import division
from builtins import str
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.core.time_pdf import TimePDF

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 50.

llh_energy = injection_energy

zero_bound = {
    "LLH Energy PDF": llh_energy,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": False
}

negative_bound = {
    "LLH Energy PDF": llh_energy,
    "Fit Gamma?": True,
    "Flare Search?": False,
    "Fit Negative n_s?": True
}

name = "benchmarks/negative_n_s_time-dep/"

sindecs = np.linspace(0.90, -0.90, 13)
sindecs = np.linspace(0.5, -0.5, 3)
# sindecs = [0.0]

offsets = np.linspace(50, 600, 12)
# offsets = np.linspace(50, 300, 6)

length = 100.


analyses = dict()

for sindec in sindecs:
    cat_path = ps_catalogue_name(sindec)

    decname = name + "sindec=" + '{0:.2f}'.format(sindec) + "/"

    src_res = dict()

    for i, raw_llh_kwargs in enumerate([zero_bound, negative_bound]):

        label = ["Zero-bound", "Negative-bound"][i]

        time_res = dict()

        for offset in offsets:

            full_name = decname + label + "_" + str(offset) + "/"

            time = {
                "Name": "FixedRefBox",
                "Fixed Ref Time (MJD)": 56300.,
                "Pre-Window": 0.,
                "Post-Window": float(offset)
            }

            injection_time = time

            inj_kwargs = {
                "Injection Energy PDF": injection_energy,
                "Injection Time PDF": injection_time,
                "Poisson Smear?": True,
            }

            llh_kwargs = dict(raw_llh_kwargs)
            llh_kwargs["LLH Time PDF"] = time

            scale = flux_to_k(reference_sensitivity(sindec)) * max_window

            mh_dict = {
                "name": full_name,
                "datasets": custom_dataset(ps_v002_p01[-1:], np.load(cat_path),
                                           llh_kwargs["LLH Time PDF"]),
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 100,
                "n_steps": 20
            }

            analysis_path = analysis_dir + full_name

            try:
                os.makedirs(analysis_path)
            except OSError:
                pass

            pkl_file = analysis_path + "dict.pkl"

            with open(pkl_file, "wb") as f:
                Pickle.dump(mh_dict, f)

            rd.submit_to_cluster(pkl_file, n_jobs=100)

            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=5)

            time_res[offset] = mh_dict

        src_res[label] = time_res

    analyses[sindec] = src_res

rd.wait_for_cluster()

for sindec, src_res in analyses.items():

    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), colspan=3, rowspan=3)

    plt.title("Sensitivity with Time Integration")

    all_sens = []
    all_fracs = []

    cols = ["b", "orange", "g"]

    frac_over = []

    for i, (label, method_res) in enumerate(src_res.items()):

        sens = []
        fracs = []
        disc_pots = []
        ts_median = []

        for offset in offsets:

            rh_dict = method_res[offset]

            try:
                rh = ResultsHandler(rh_dict)

                injection_time = rh_dict["inj kwargs"]["Injection Time PDF"]

                inj_time = 0.

                cat = np.load(rh_dict["catalogue"])

                for season in rh_dict["datasets"]:
                    time = TimePDF.create(injection_time, season)
                    inj_time += np.mean([
                        time.effective_injection_time(src) for src in cat])

                sens.append(rh.sensitivity*inj_time)
                disc_pots.append(rh.disc_potential*inj_time)
                ts_median.append(rh.bkg_median)

                if label == "Zero-bound":

                    frac_over.append(rh.frac_over)

            except EOFError:
                sens.append(np.nan)
                disc_pots.append(np.nan)

            fracs.append(offset)

        mask = ~np.isnan(sens)

        fracs = np.array(fracs)
        sens = np.array(sens)
        disc_pots = np.array(disc_pots)

        ax1.plot(fracs[mask], sens[mask], label=label, color=cols[i])

        all_sens.append(sens)
        all_fracs.append(fracs)

    ax1.grid(True, which='both')
    yticks = ax1.yaxis.get_major_ticks()
    yticks[1].label1.set_visible(False)

    ax1.set_ylabel(r"Integrated Flux [ GeV$^{-1}$ cm$^{-2}$ ]",
                   fontsize=12)

    ax2 = plt.subplot2grid((6, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

    ax3 = plt.subplot2grid((6, 1), (4, 0), colspan=3, rowspan=1, sharex=ax1)

    ax4 = plt.subplot2grid((6, 1), (5, 0), colspan=3, rowspan=1, sharex=ax1)

    ratios = np.array(all_sens[1]) / np.array(all_sens[0])

    mask = ~np.isnan(ratios)

    ax2.scatter(fracs[mask], ratios[mask], color="red")
    ax2.plot(fracs[mask], ratios[mask], color="red")
    ax2.set_ylabel(r"Ratio", fontsize=12)

    ax2.grid(True)

    ax3.plot(fracs, ts_median, color="green")

    ax3.set_ylabel(r"$\lambda_{med}$")
    ax3.grid(True, which='both')

    ax4.plot(fracs, frac_over, color="purple")
    ax4.set_ylabel(r"Overfluctuations")
    ax4.grid(True, which='both')


    ax4.set_xlabel(r"Length of window (day)", fontsize=12)

    xticklabels = ax1.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    plt.subplots_adjust(hspace=0.001)

    # ax1.set_xscale("log")
    # ax1.set_xlim(0, 1.0)

    ax1.legend(loc='upper left', fancybox=True, framealpha=1.)

    savename = plot_output_dir(name) + "sindec=" + '{0:.2f}'.format(sindec) + \
               "_ratio.pdf"

    try:
        os.makedirs(os.path.dirname(savename))
    except OSError:
        pass

    fig.set_size_inches((6, 8))

    # plt.tight_layout()

    plt.savefig(savename)
    plt.close()

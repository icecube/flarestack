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

analyses = dict()

# Start and end time of flare in MJD
t_start = 55740.00
t_end = 55840.00

max_window = float(t_end - t_start)

# Initialise Injectors/LLHs

llh_energy = {
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

fit_weights = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    # "Fit Negative n_s?": True,
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

# flare = {
#     "LLH Energy PDF": llh_energy,
#     "LLH Time PDF": llh_time,
#     "Fit Gamma?": True,
#     "Flare Search?": True,
#     "Fit Negative n_s?": False
# }

max_window = 100

max_window_s = max_window * 60 * 60 * 24


gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9]
# gammas = [2.0, 2.3]
# gammas = [1.99, 2.0, 2.02]
# gammas = [2.5, 2.7, 2.9]

cat_res = dict()

cats = ["gold", "jetted"]
# cats = ["jetted"]

for cat in cats:

    name = "analyses/tde/compare_spectral_indices/" + cat + "/"

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    src_res = dict()

    closest_src = np.sort(catalogue, order="Distance (Mpc)")[0]

    # lengths = [0.5 * max_window]

    for i, llh_kwargs in enumerate([fixed_weights_negative,
                                    fixed_weights,
                                    fit_weights
                                    ]):
        label = ["Fixed Weights (Negative n_s)", "Fixed Weights",
                 "Fit Weights", "Flare Search", ][i]
        f_name = ["fixed_weights_neg", "fixed_weights",
                  "fit_weights", "flare"][i]

        flare_name = name + f_name + "/"

        res = dict()

        for gamma in gammas:

            full_name = flare_name + str(gamma) + "/"

            injection_time = dict(llh_time)

            injection_energy = dict(llh_energy)
            # injection_energy["E Min"] = 10000
            injection_energy["Gamma"] = gamma

            inj_kwargs = {
                "Injection Energy PDF": injection_energy,
                "Injection Time PDF": injection_time,
                "Poisson Smear?": True,
            }

            scale = flux_to_k(skylab_7year_sensitivity(
                np.sin(closest_src["dec"]), gamma=gamma
            ) * 50)

            # print scale

            mh_dict = {
                "name": full_name,
                "datasets": ps_7year[-2:-1],
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 10,
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

            rd.submit_to_cluster(pkl_file, n_jobs=200)
            #
            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=200)
            # mh.clear()
            res[gamma] = mh_dict

        src_res[label] = res

    cat_res[cat] = src_res

rd.wait_for_cluster()

for (cat, src_res) in cat_res.iteritems():

    name = "analyses/tde/compare_spectral_indices/" + cat + "/"

    sens = [[] for _ in src_res]
    sens_livetime = [[] for _ in src_res]
    fracs = [[] for _ in src_res]
    disc_pots = [[] for _ in src_res]
    disc_pots_livetime = [[] for _ in src_res]
    sens_e = [[] for _ in src_res]
    disc_e = [[] for _ in src_res]

    labels = []

    for i, (f_type, res) in enumerate(sorted(src_res.iteritems())):
        for (gamma, rh_dict) in sorted(res.iteritems()):
            try:
                rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                    rh_dict["catalogue"])

                # The uptime can noticeably devaiate from 100%
                injection_time = rh_dict["inj kwargs"]["Injection Time PDF"]

                inj_time = 0.

                for season in rh_dict["datasets"]:
                    time = TimePDF.create(injection_time, season)
                    inj_time += time.effective_injection_time(catalogue)

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj kwargs"]["Injection Energy PDF"])

                key = "Total Fluence (GeV^{-1} cm^{-2} s^{-1})"

                e_key = "Total Luminosity (erg/s)"

                sens[i].append(astro_sens[key] * max_window_s)
                disc_pots[i].append(astro_disc[key] * max_window_s)
                sens_livetime[i].append(astro_sens[key] * inj_time)
                disc_pots_livetime[i].append(astro_disc[key] * inj_time)

                sens_e[i].append(astro_sens[e_key] * max_window_s)
                disc_e[i].append(astro_disc[e_key] * max_window_s)

                fracs[i].append(gamma)

                print rh.sensitivity * inj_time

            except OSError:
                pass

        labels.append(f_type)
        # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    for j, s in enumerate([sens, sens_livetime, sens_e]):

        d = [disc_pots, disc_pots_livetime, disc_e][j]

        for k, y in enumerate([s, d]):

            plt.figure()
            ax1 = plt.subplot(111)

            cols = ["r", "g", "b", "orange"]
            linestyle = ["-", "--"][k]

            for i, f in enumerate(fracs):
                plt.plot(f, y[i], label=labels[i], linestyle=linestyle,
                         color=cols[i])

            label = ["", "(Livetime-adjusted)", "energy"][j]

            y_label = [r"Fluence [GeV cm$^{-2}$]", r"Fluence [GeV cm$^{-2}$]",
                       r"Total Isotropic-Equivalent $E_{\nu}$ (erg)"]

            ax1.grid(True, which='both')
            ax1.set_ylabel(y_label[j], fontsize=12)
            ax1.set_xlabel(r"Gamma")
            ax1.set_yscale("log")
            ax1.set_ylim(0.95 * min([min(x) for x in y]),
                         1.1 * max([max(x) for x in y]))

            plt.title("Time-Integrated Emission in " + str(int(max_window)) +
                      " day window")

            ax1.legend(loc='upper right', fancybox=True, framealpha=1.)
            plt.tight_layout()
            plt.savefig(plot_output_dir(name) + "/spectral_index_" + label +
                        "_" + ["sens", "disc"][k] + ".pdf")
            plt.close()
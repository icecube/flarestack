import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7_year import ps_7year
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
from utils.skylab_reference import skylab_7year_sensitivity
from cluster import run_desy_cluster as rd
import matplotlib as mpl
mpl.use('Agg')

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

new_style = {                      # setup matplotlib to use latex for output
    # "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    # "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    }
mpl.rcParams.update(new_style)

import matplotlib.pyplot as plt
from core.time_PDFs import TimePDF

name = "analyses/tde/compare_flare_to_box/"

analyses = dict()

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

# Start and end time of flare in MJD
t_start = 55752.00
t_end = 55852.00

cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
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

# lengths = np.array(
#     sorted([00.03, 0.05] + list(np.linspace(0.0, 1.0, 4)))[1:])
#  * max_window

lengths = np.logspace(-2, 0, 9) * max_window
# lengths = np.logspace(-2, 0, 2) * max_window

# lengths = [0.5 * max_window]

for i, llh_kwargs in enumerate([
    no_flare, no_flare_negative,
                                flare_with_energy]):

    label = ["Time-Integrated", "Time-Integrated (negative n_s)",
             "Cluster Search"][i]
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

        scale = flux_to_k(skylab_7year_sensitivity(np.sin(catalogue["dec"]))
                          * (50 * max_window / flare_length))

        # print scale, scale * flare_length / max_window

        mh_dict = {
            "name": full_name,
            "datasets": ps_7year[-2:-1],
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

        # print "Injecting for", flare_length, "Livetime", inj_time/(60.*60.*24.)

        # rd.submit_to_cluster(pkl_file, n_jobs=5000)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=50)
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

    if f_type!="Time-Integrated (negative n_s)":

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

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj kwargs"]["Injection Energy PDF"])

                key = "Total Fluence (GeV^{-1} cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                sens_livetime[i].append(astro_sens[key] * inj_time)
                disc_pots_livetime[i].append(astro_disc[key] * inj_time)

                # sens[i].append(rh.sensitivity * float(length) * 60 * 60 * 24)
                # disc_pots[i].append(rh.disc_potential *
                #                     float(length) * 60 * 60 * 24)
                # sens_livetime[i].append(rh.sensitivity * inj_time)
                # disc_pots_livetime[i].append(rh.disc_potential * inj_time)
                fracs[i].append(inj_time/(60 * 60 * 24))

                # raw_input("prompt")

            except OSError:
                pass

            except KeyError:
                pass

            except EOFError:
                pass

        labels.append(f_type)
    # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

for j, s in enumerate([sens_livetime]):

    d = [disc_pots_livetime][j]

    for k, y in enumerate([s, d]):

        plt.figure()
        ax1 = plt.subplot(111)

        cols = ["#F79646", "#00A6EB", "r", "g",]
        linestyle = ["-", "-"][k]

        for i, f in enumerate(fracs):
            if len(f) > 0:
                plt.plot(f, y[i], label=labels[i], linestyle=linestyle,
                         color=cols[i])

        # ax1.grid(True, which='both')
        # ax1.semilogy(nonposy='clip')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]",
                       fontsize=12)
        ax1.set_xlabel(r"Neutrino Flare Length (days)")
        # ax1.set_xscale("log")
        ax1.set_ylim(0.95 * min([min(x) for x in y if len(x) > 0]),
                     1.1 * max([max(x) for x in y if len(x) > 0]))

        plt.title(["Sensitivity", "Discovery Potential"][k] + " for " + \
                  str(int(max_window)) + " day search window")

        ax1.legend(loc='upper left', fancybox=True, framealpha=0.)
        plt.savefig(plot_output_dir(name) + "/flare_vs_box_" +
                    ["sens", "disc"][k] + ".pdf")
        plt.close()

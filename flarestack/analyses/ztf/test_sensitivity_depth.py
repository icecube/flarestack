import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
import datetime
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.injector import Injector
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.utils.simulate_catalogue import simulate_transient_catalogue
from flarestack.analyses.ztf.sn_cosmology import get_sn_type_rate
from flarestack.shared import make_analysis_pickle

name_root = "analyses/ztf/depth/"


# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

pre_window = 5.

# Use a source that is constant in time

injection_time = {
    "Name": "Steady",
    "Pre-Window": 0.,
    "Post-Window": 100.
}

# Use a source with a spectral index of -2, with an energy range between
# 100 GeV and 10 Pev (10**7 GeV).

injection_gamma = 2.0

injection_energy = {
    "Name": "Power Law",
    "Gamma": injection_gamma,
    "E Min": 10 ** 2,
    "E Max": 10 ** 7
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

# Look for a source that is constant in time

llh_time = dict(inj_kwargs["Injection Time PDF"])

# Try to fit a power law to the data

llh_energy = {
    "Name": "Power Law",
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
}

raw_mh_dict = {
    "datasets": ps_7year[-2:],
    "inj kwargs": inj_kwargs,
    "llh kwargs": llh_kwargs,
    "n_trials": 5,
    "n_steps": 30
}

# Now simulate catalogue

sn_types = ["SNIIn"]

res_dict = dict()

for sn in sn_types:

    sn_dict = dict()

    base_name = name_root + sn + "/"
    rate = get_sn_type_rate(sn_type=sn)
    all_cat_names = simulate_transient_catalogue(raw_mh_dict, rate,
                                                 cat_name="random_" + sn)

    for (sky, cat_names) in all_cat_names.iteritems():

        sky_dict = dict()

        sky_name = base_name + sky + "/"

        for i, cat in enumerate(cat_names):

            # Skip if catalogue is empty
            if len(np.load(cat)) == 0:
                continue

            # Skips if already tested:
            elif float(len(np.load(cat))) in sky_dict.keys():
                continue

            name = sky_name + os.path.basename(cat)[:-4] + "/"

            # cat_path = catalogue_dir + "TDEs/TDE_silver_catalogue.npy"
            # catalogue = np.load(cat_path)

            scale = flux_to_k(reference_sensitivity(
                np.sin(0.0))) * 160

            mh_dict = dict(raw_mh_dict)
            mh_dict["name"] = name
            mh_dict["catalogue"] = cat
            mh_dict["scale"] = scale

            pkl_file = make_analysis_pickle(mh_dict)

            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(scale=scale, n_steps=mh_dict["n_steps"],
            #                n_trials=mh_dict["n_trials"])
            # mh.clear()
            # rd.submit_to_cluster(pkl_file, n_jobs=100)

            sky_dict[float(len(np.load(cat)))] = mh_dict

        sn_dict[sky] = sky_dict

    res_dict[sn] = sn_dict

rd.wait_for_cluster()

for (sn, sn_dict) in res_dict.iteritems():

    savedir_sn = plot_output_dir(name_root) + sn + "/"
    for (sky, sky_dict) in sn_dict.iteritems():

        savedir = savedir_sn + sky + "/"

        sens = []
        sens_e = []
        disc = []
        disc_e = []
        n = []

        dist = []

        for (n_cat, rh_dict) in sorted(sky_dict.iteritems()):
            rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                rh_dict["catalogue"], show_inj=True)

            inj_time = pre_window * 60 * 60 * 24

            astro_sens, astro_disc = rh.astro_values(
                rh_dict["inj kwargs"]["Injection Energy PDF"])

            energy_pdf = EnergyPDF.create(rh_dict["inj kwargs"]["Injection Energy PDF"])
            #
            # raw_input("prompt")

            key = "Total Fluence (GeV cm^{-2} s^{-1})"

            e_key = "Mean Luminosity (erg/s)"

            sens.append(astro_sens[key] * inj_time)
            disc.append(astro_disc[key] * inj_time)

            sens_e.append(astro_sens[e_key] * inj_time)
            disc_e.append(astro_disc[e_key] * inj_time)

            cat = np.load(rh_dict["catalogue"])
            dist.append(max(cat["Distance (Mpc)"]))
            n.append(float(len(cat)))

            print "N_sources", len(cat)

        try:
            os.makedirs(os.path.dirname(savedir))
        except OSError:
            pass

        n_av = 3

        av_dist = [np.median(dist[i:i+n_av]) for i in range(len(dist) - n_av)]
        av_sens = [np.mean(sens[i:i+n_av]) for i in range(len(dist) - n_av)]
        av_sens_e = [np.mean(sens_e[i:i+n_av]) for i in range(len(dist) - n_av)]

        label = str(n_av) + "-point rolling average"

        plt.figure()
        ax1 = plt.subplot(111)
        # ax2 = ax1.twinx()
        ax1.plot(dist, sens)
        ax1.plot(av_dist, av_sens, linestyle=":", label=label)
        plt.legend()
        plt.xlabel("Maximum Distance (Mpc)")
        ax1.set_ylabel(r"Time-Integrated Flux [ GeV$^{-1}$ cm$^{-2}$]")
        plt.savefig(savedir + "detected_flux.pdf")
        plt.close()

        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(dist, sens_e)
        ax1.plot(av_dist, av_sens_e, linestyle=":", label=label)
        # ax1.plot(dist, disc_e)
        plt.legend()
        plt.xlabel("Maximum Distance (Mpc)")
        ax1.set_ylabel(r"Energy per source (erg)")
        plt.savefig(savedir + "e_per_source.pdf")
        plt.close()

        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(dist, n)
        plt.yscale("log")
        plt.xlabel("Maximum  Distance (Mpc)")
        ax1.set_ylabel("Cumulative Sources")
        plt.savefig(savedir + "n_source.pdf")
        plt.close()


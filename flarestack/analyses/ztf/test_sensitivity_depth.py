import numpy as np
import os
from flarestack.core.results import ResultsHandler
from flarestack.core.minimisation import MinimisationHandler
from flarestack.data.icecube.gfu.gfu_v002_p04 import gfu_v002_p04
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.utils.simulate_catalogue import simulate_transient_catalogue
from flarestack.analyses.ccsn.sn_cosmology import get_sn_type_rate
from flarestack.utils.catalogue_loader import load_catalogue

name_root = "analyses/ztf/depth/"

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

post_window = 100.

# Use a source that is constant in time

# injection_time = {
#     "Name": "Box",
#     "Pre-Window": 0.,
#     "Post-Window": post_window
# }
injection_time = {
    "time_pdf_name": "Steady"
}

# Use a source with a spectral index of -2, with an energy range between
# 100 GeV and 10 Pev (10**7 GeV).

injection_gamma = 2.0

injection_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": injection_gamma,
    "e_min_gev": 10 ** 2,
    "e_max_gev": 10 ** 7
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_kwargs = {
    "injection_energy_pdf": injection_energy,
    "injection_time_pdf": injection_time,
    "poisson_smear_bool": True,
}

# Look for a source that is constant in time

llh_time = dict(inj_kwargs["injection_time_pdf"])

# Try to fit a power law to the data

llh_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": injection_gamma
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs = {
    "llh_name": "standard_matrix",
    "llh_energy_pdf": llh_energy,
    "llh_time_pdf": llh_time,
}

raw_mh_dict = {
    "mh_name": "large_catalogue",
    "datasets": [IC86_1_dict],
    "inj_dict": inj_kwargs,
    "llh_dict": llh_kwargs,
    "n_trials": 1,
    "n_steps": 15
}

# Now simulate catalogue

sn_types = ["IIn"]

res_dict = dict()

for sn in sn_types:

    sn_dict = dict()

    base_name = name_root + sn + "/"
    rate = get_sn_type_rate(sn_type=sn)
    all_cat_names = simulate_transient_catalogue(raw_mh_dict, rate,
                                                 cat_name="random_" + sn,
                                                 )

    # print [x for x in all_cat_names.itervalues()]
    # raw_input("prompt")

    for (sky, cat_names) in all_cat_names.iteritems():

        sky_dict = dict()

        sky_name = base_name + sky + "/"

        for i, cat_name in enumerate(cat_names[:-5]):
            cat = load_catalogue(cat_name)

            n_cat = float(len(cat))

            # Skip if catalogue is empty
            if len(cat) == 0:
                continue

            # Skips if already tested:
            if n_cat in sky_dict.keys():
                continue

            name = sky_name + os.path.basename(cat_name)[:-4] + "/"

            cat_scale = np.sum(cat["distance_mpc"]**-2 /
                               min(cat["distance_mpc"])**-2)

            # cat_scale = min(cat["distance_mpc"])**2/\
            #             np.sum(cat["distance_mpc"]**-2)

            scale = flux_to_k(reference_sensitivity(
                np.sin(cat[0]["dec_rad"]))) * cat_scale * 25.

            if "full" in cat_name:
                scale *= 0.5

            # from flarestack.analyses.agn_cores.shared_agncores import agncores_cat_dir
            # cat_name = agncores_cat_dir + "radioloud_radioselected_100brightest_srcs.npy"

            mh_dict = dict(raw_mh_dict)
            mh_dict["name"] = name
            mh_dict["catalogue"] = cat_name
            mh_dict["scale"] = scale

            pkl_file = make_analysis_pickle(mh_dict)

            mh = MinimisationHandler.create(mh_dict)
            mh.iterate_run(scale=scale, n_steps=mh_dict["n_steps"],
                           n_trials=mh_dict["n_trials"])
            # mh_power_law.clear()
            # rd.submit_to_cluster(pkl_file, n_jobs=100)

            sky_dict[n_cat] = mh_dict

        sn_dict[sky] = sky_dict

    res_dict[sn] = sn_dict

# raw_input("prompt")

rd.wait_for_cluster()

for (sn, sn_dict) in res_dict.iteritems():

    savedir_sn = plot_output_dir(name_root) + sn + "/"
    for (sky, sky_dict) in sn_dict.iteritems():

        if sky == "Northern":

            savedir = savedir_sn + sky + "/"

            sens = []
            sens_e = []
            disc = []
            disc_e = []
            n = []

            dist = []

            for (n_cat, rh_dict) in sorted(sky_dict.iteritems()):
                rh = ResultsHandler(rh_dict)

                inj_time = post_window * 60 * 60 * 24

                astro_sens, astro_disc = rh.astro_values(
                    rh_dict["inj_dict"]["injection_energy_pdf"])

                energy_pdf = EnergyPDF.create(
                    rh_dict["inj_dict"]["injection_energy_pdf"])
                #
                # raw_input("prompt")

                key = "Total Fluence (GeV cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                sens.append(astro_sens[key] * inj_time)
                disc.append(astro_disc[key] * inj_time)

                sens_e.append(astro_sens[e_key] * inj_time)
                disc_e.append(astro_disc[e_key] * inj_time)

                cat = np.load(rh_dict["catalogue"])
                dist.append(max(cat["distance_mpc"]))
                n.append(float(len(cat)))

                print "N_sources", len(cat)

            try:
                os.makedirs(os.path.dirname(savedir))
            except OSError:
                pass

            print dist, sens_e, n, disc_e

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


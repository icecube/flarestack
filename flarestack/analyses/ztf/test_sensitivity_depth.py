import numpy as np
import os
from flarestack.core.results import ResultsHandler
from flarestack.core.minimisation import MinimisationHandler
from flarestack.data.icecube.gfu.gfu_v002_p04 import gfu_v002_p04
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import plot_output_dir, flux_to_k, \
    make_analysis_pickle, k_to_flux
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.utils.simulate_catalogue import simulate_transient_catalogue
from flarestack.analyses.ccsn.sn_cosmology import get_sn_type_rate
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.utils.neutrino_astronomy import calculate_astronomy
from flarestack.utils.deus_ex_machina import DeusExMachina

name_root = "analyses/ztf/depth/"

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

post_window = 100.

# Use a source that is constant in time

injection_time = {
    "Name": "Box",
    "Pre-Window": 0.,
    "Post-Window": post_window
}
#injection_time = {
#    "time_pdf_name": "Steady"
#}

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
    "n_trials": 5,
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
                                                 n_entries=25,
                                                 )

    # print [x for x in all_cat_names.itervalues()]
    # raw_input("prompt")

    for (sky, cat_names) in all_cat_names.iteritems():

        sky_dict = dict()

        sky_name = base_name + sky + "/"

        for i, cat_name in enumerate(cat_names):

            n_cat = float(len(np.load(cat_name)))

            if sky not in ["Northern"]:
                continue

            # Skip if catalogue is empty
            if n_cat == 0:
                continue

            # Skips if already tested:
            if n_cat in sky_dict.keys():
                continue

            cat = load_catalogue(cat_name)

            name = sky_name + os.path.basename(cat_name)[:-4] + "/"

            cat_scale = np.sum(cat["distance_mpc"]**-2 /
                               min(cat["distance_mpc"])**-2)

            # cat_scale = min(cat["distance_mpc"])**2/\
            #             np.sum(cat["distance_mpc"]**-2)

            scale = flux_to_k(reference_sensitivity(
                np.sin(cat[0]["dec_rad"]))) * cat_scale * 15.

            if "full" in cat_name:
                scale *= 0.5

            # from flarestack.analyses.agn_cores.shared_agncores import agncores_cat_dir
            # cat_name = agncores_cat_dir + "radioloud_radioselected_100brightest_srcs.npy"

            mh_dict = dict(raw_mh_dict)
            mh_dict["name"] = name
            mh_dict["catalogue"] = cat_name
            mh_dict["scale"] = scale

            pkl_file = make_analysis_pickle(mh_dict)

            # if n_cat < 1000.:
            #     rd.submit_to_cluster(pkl_file, n_jobs=100)
            # print cat_name
            # mh = MinimisationHandler.create(mh_dict)
            #     print mh.run_trial(scale=scale)
            #     raw_input("prompt")

            # mh.run_trial(scale=scale)
            # mh_power_law.clear()
            # rd.submit_to_cluster(pkl_file, n_jobs=100)
            if n_cat < 1000.:
                sky_dict[n_cat] = mh_dict

        sn_dict[sky] = sky_dict

    res_dict[sn] = sn_dict

# raw_input("prompt")

rd.wait_for_cluster()

dem = DeusExMachina([IC86_1_dict], inj_kwargs)

for (sn, sn_dict) in res_dict.iteritems():

    savedir_sn = plot_output_dir(name_root) + sn + "/"
    for (sky, sky_dict) in sn_dict.iteritems():

        if sky == "Northern":
        # if True:

            savedir = savedir_sn + sky + "/"

            sens = []
            sens_e = []
            disc = []
            disc_e = []
            disc_25 = []
            disc_25_e = []
            guess_disc = []
            guess_disc_e = []
            n = []

            dist = []

            for (n_cat, rh_dict) in sorted(sky_dict.iteritems())[1:9]:
                inj_time = post_window * 60 * 60 * 24

                key = "Total Fluence (GeV cm^{-2} s^{-1})"

                e_key = "Mean Luminosity (erg/s)"

                # rh = ResultsHandler(rh_dict)
                # #
                # # #
                # astro_sens, astro_disc = rh.astro_values(
                #     rh_dict["inj_dict"]["injection_energy_pdf"])
                # #
                # # energy_pdf = EnergyPDF.create(
                # #     rh_dict["inj_dict"]["injection_energy_pdf"])
                # #
                # # raw_input("prompt")
                #
                # disc_convert = rh.disc_potential_25/rh.disc_potential
                # #
                # sens.append(astro_sens[key] * inj_time)
                # disc.append(astro_disc[key] * inj_time)
                # #
                # # # print astro_disc[key], rh.disc_potential, guess_disc
                # # # raw_input("prompt")
                # #
                # disc_25.append(astro_disc[key] * inj_time * disc_convert)
                # #
                # sens_e.append(astro_sens[e_key] * inj_time)
                # disc_e.append(astro_disc[e_key] * inj_time)
                # disc_25_e.append(astro_disc[e_key] * inj_time * disc_convert)

                cat = load_catalogue(rh_dict["catalogue"])

                guess = k_to_flux(
                    dem.guess_discovery_potential(rh_dict["catalogue"])
                )

                astro_guess = calculate_astronomy(
                    guess, rh_dict["inj_dict"]["injection_energy_pdf"], cat
                )

                guess_disc.append(astro_guess[key] * inj_time)
                guess_disc_e.append(astro_guess[e_key] * inj_time)

                dist.append(max(cat["distance_mpc"]))
                n.append(float(len(cat)))

                print n_cat


            try:
                os.makedirs(os.path.dirname(savedir))
            except OSError:
                pass

            pairs = [
                (guess_disc, guess_disc_e),
                # (sens, sens_e),
                # (disc, disc_e),
                # (disc_25, disc_25_e)

            ]

            for j, (vals, vals_e) in enumerate(pairs):

                label = ["guess_disc", "sensitivity", "disc", "disc_25"][j]

                plt.figure()
                ax1 = plt.subplot(111)
                # ax2 = ax1.twinx()
                ax1.plot(dist, vals)
                # ax1.plot(av_dist, av_sens, linestyle=":", label=label)
                # plt.legend()
                plt.xlabel("Maximum Distance (Mpc)")
                ax1.set_ylabel(r"Time-Integrated Flux [ GeV$^{-1}$ cm$^{-2}$]")
                plt.savefig(savedir + label + "_detected_flux.pdf")
                plt.close()

                vals_e = np.array(vals_e)

                base_mask = ~np.isnan(vals_e)
                mask = vals_e[base_mask] > 0.

                dists = np.array(np.log(dist))[base_mask][mask]
                log_e = np.log(vals_e[base_mask][mask])

                z = np.polyfit(dists, log_e, 1)

                def f(x):
                    return np.exp(z[1] + (z[0] * np.log(x)))

                dist_range = np.linspace(min(dist), max(dist), 1e3)

                plt.figure()
                ax1 = plt.subplot(111)
                ax1.scatter(np.exp(dists), vals_e[base_mask][mask])
                # ax1.plot(dist_range, f(dist_range))
                # ax1.plot(av_dist, av_sens_e, linestyle=":", label=label)
                # ax1.plot(dist, disc_e)
                # plt.legend()
                plt.xlabel("Maximum Distance (Mpc)")
                ax1.set_ylabel(r"Energy per source (erg)")
                plt.savefig(savedir + label + "_e_per_source.pdf")
                plt.close()

                plt.figure()
                ax1 = plt.subplot(111)
                ax1.scatter(np.exp(dists), vals_e[base_mask][mask])
                # ax1.plot(dist_range, f(dist_range))
                # ax1.plot(av_dist, av_sens_e, linestyle=":", label=label)
                # ax1.plot(dist, disc_e)
                # plt.legend()
                plt.xlabel("Maximum Distance (Mpc)")
                ax1.set_ylabel(r"Energy per source (erg)")
                plt.yscale("log")
                plt.xscale("log")
                plt.savefig(savedir + label + "_log_e_per_source.pdf")
                plt.close()

                plt.figure()
                ax1 = plt.subplot(111)
                ax1.plot(dist, n)
                plt.yscale("log")
                plt.xlabel("Maximum  Distance (Mpc)")
                ax1.set_ylabel("Cumulative Sources")
                plt.savefig(savedir + label + "_n_source.pdf")
                plt.close()

from __future__ import division
import numpy as np
import os
from flarestack.shared import plot_output_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
import matplotlib.pyplot as plt
from flarestack.utils.asimov_estimator import AsimovEstimator
from flarestack import analyse, MinimisationHandler
from flarestack.data.simulate.simcube import simcube_dataset
from flarestack.data.public import icecube_ps_3_year

name = "analyses/simcube/verify_sensitivity"

# Initialise Injectors/LLHs

injection_gamma = 2.0

injection_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": injection_gamma,
}

injection_time = {
    "time_pdf_name": "steady",
}

llh_time = injection_time

inj_dict = {
    "injection_energy_pdf": injection_energy,
    "injection_time_pdf": injection_time,
}

llh_energy = injection_energy

llh_dict = {
    "name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_time_pdf": llh_time,
}

sindecs = np.linspace(0.75, 0.0, 4)

datasets = [
    ("IceCube (One Year", icecube_ps_3_year.get_seasons("IC86-2012")),
    # ("Simcube (One year)", simcube_dataset.get_seasons()),
]

# plt.figure()
# ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
# ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)
# refs = reference_7year_discovery_potential(sindecs, injection_gamma)
#
# ax1.plot(sindecs, refs, label=r"7-year Point Source analysis", color="k")

for i, (label, dataset) in enumerate(datasets):
    for sindec in sindecs:
        cat_path = ps_catalogue_name(sindec)

        mh_dict = {
            "name": name + "/sindec=" + "{0:.2f}".format(sindec) + "/",
            "mh_name": "fixed_weights",
            "datasets": dataset,
            "catalogue": cat_path,
            "llh_dict": llh_dict,
            "inj_dict": inj_dict,
            "n_trials": 10,
            "n_steps": 15,
        }

        mh_dict["scale"] = 1.0

        mh = MinimisationHandler.create(mh_dict)
        # print(mh.simulate_and_run(0.))
        print(mh.guess_scale())

        # analyse(mh_dict, n_cpu=2)

        input("?")

    #     disc_pots = np.array(disc_pots)
    #
    #     plot_range = np.linspace(-0.99, 0.99, 1000)
    #
    #     ax1.plot(sindecs[mask], disc_pots, color=color,
    #              label="Flarestack Estimation ({0})".format(sample_name))
    #
    #     disc_ratios = np.array(disc_pots)/refs[mask]
    #     print(disc_ratios)
    #     print("Range:", max(disc_ratios)/min(disc_ratios))
    #
    #     ax2.scatter(sindecs[mask], disc_ratios, color=color)
    #     ax2.plot(sindecs[mask], disc_ratios, color=color, linestyle="--")
    #
    # ax1.set_xlim(xmin=-1., xmax=1.)
    # # ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
    # ax1.grid(True, which='both')
    # ax1.semilogy(nonposy='clip')
    # ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
    #                fontsize=12)
    #
    # ax2.set_ylabel(r"ratio", fontsize=12)
    # ax2.set_xlabel(r"sin($\delta$)", fontsize=12)
    # #
    # ax1.set_xlim(xmin=-1.0, xmax=1.0)
    # # ax2.set_ylim(ymin=0.5, ymax=1.5)
    # ax2.grid(True)
    # xticklabels = ax1.get_xticklabels()
    # plt.setp(xticklabels, visible=False)
    # plt.subplots_adjust(hspace=0.001)
    #
    # plt.suptitle('Point Source Discovery Potential')
    #
    # # ratio_interp = interp1d(sindecs, sens_ratios)
    # #
    # # interp_range = np.linspace(np.min(sindecs),
    # #                            np.max(sindecs), 1000)
    #
    # # ax1.plot(
    # #     interp_range,
    # #     reference_sensitivity(interp_range)*ratio_interp(interp_range),
    # #     color='red', linestyle="--", label="Ratio Interpolation")
    #
    # ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

    # save_dir = plot_output_dir(name)
    #
    # try:
    #     os.makedirs(save_dir)
    # except OSError:
    #     pass
    #
    # title = ["/PSDisc.pdf", "/IC86_1.pdf"][i]
    #
    # plt.savefig(save_dir + title)
    # plt.close()

"""Script to replicate unblinding of the neutrino flare found for the source
TXS 0506+056, as described in https://arxiv.org/abs/1807.08794.
"""
import os
import matplotlib.pyplot as plt
from flarestack.core.results import ResultsHandler
import numpy as np
from flarestack.data.icecube.public.all_sky_point_source.all_sky_3_year \
    import ps_3_year
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_3_systematic_set
from flarestack.cluster import analyse, wait_cluster
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.asimov_estimator import AsimovEstimator
from flarestack.shared import plot_output_dir, flux_to_k


# Initialise Injectors/LLHs

season = "IC86-2011"

# Shared

llh_time = {
    "time_pdf_name": "Steady"
}

llh_energy = {
    "energy_pdf_name": "PowerLaw",
}

llh_dict = {
    # "llh_name": "spatial",
    "llh_time_pdf": llh_time,
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy
}

inj_time = llh_time
inj_energy_pdf = {
    "energy_pdf_name": "PowerLaw",
    "gamma": 2
}

inj_dict = {
    "injection_dataset": ps_3_systematic_set.get_seasons(season),
    "injection_time_pdf": inj_time,
    "injection_energy_pdf": inj_energy_pdf
}

datasets = [
    ps_3_systematic_set.get_seasons(season),
    ps_3_year.get_seasons(season)
]

base_name = "general/compare_public_data_sensitivity/"

sindecs = np.linspace(0.90, -0.90, 13)

ae = AsimovEstimator(ps_3_systematic_set.get_seasons(season), inj_dict)

all_res = dict()


for i, dataset in enumerate(datasets):
    key = ["public", "internal"][i]

    label = ["Public (Effective Area)", "Internal (MC)"][i]

    name = base_name + key + "/"

    dataset_res = dict()

    for sindec in sindecs:
        cat_path = ps_catalogue_name(sindec)

        subname = name + "sindec=" + '{0:.2f}'.format(sindec) + "/"

        mh_dict = {
            "name": subname,
            "mh_name": "fixed_weights",
            "dataset": dataset,
            "catalogue": cat_path,
            "llh_dict": llh_dict,
            "inj_dict": inj_dict,
            "n_steps": 15,
            "n_trials": 1000
        }

        mh_dict["scale"] = flux_to_k(
            ae.guess_discovery_potential(cat_path) * 1.5)

        # analyse(mh_dict, n_cpu=24)

        dataset_res[sindec] = mh_dict

    all_res[label] = dataset_res


wait_cluster()

# Plot results

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

all_sens = []
all_disc = []


for i, (label, dataset_res) in enumerate(all_res.items()):

    sens = []
    disc_pots = []

    for rh_dict in dataset_res.values():

        rh = ResultsHandler(rh_dict)
        sens.append(rh.sensitivity)
        disc_pots.append(rh.disc_potential)

    plot_range = np.linspace(-0.99, 0.99, 1000)

    color=["orange", "blue"][i]

    ax1.plot(sindecs, sens, color=color, label=label)

    ax1.plot(
        sindecs, disc_pots, color=color, linestyle="--")

    all_sens.append(sens)
    all_disc.append(disc_pots)

ax1.set_xlim(xmin=-1., xmax=1.)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
               fontsize=12)

plt.title('Point Source Sensitivity {0}'.format(season))

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)
#
sens_ratios = np.array(all_sens[1]) / np.array(all_sens[0])
#
disc_ratios = np.array(all_disc[1]) / np.array(all_disc[0])
#
ax2.scatter(sindecs, sens_ratios, color="red")
ax2.plot(sindecs, sens_ratios, color="red")
ax2.scatter(sindecs, disc_ratios, color="k")
ax2.plot(sindecs, disc_ratios, color="k", linestyle="--")
ax2.set_ylabel(r"ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)

ax2.grid(True)
xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

savepath = plot_output_dir(base_name) + "/PS_comparison.pdf"

try:
    os.makedirs(os.path.dirname(savepath))
except OSError:
    pass

plt.savefig(savepath)
plt.close()

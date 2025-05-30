import logging
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from flarestack import MinimisationHandler, ResultsHandler, analyse, wait_cluster
from flarestack.data.icecube import nt_v005_p01
from flarestack.icecube_utils.reference_sensitivity import (
    reference_discovery_potential,
    reference_sensitivity,
)
from flarestack.shared import flux_to_k, plot_output_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name

logging.basicConfig(level=logging.INFO)

# Initialise Injectors/LLHs

injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

injection_time = {
    "time_pdf_name": "steady",
}

injection_spatial = {
    "spatial_pdf_name": "northern_tracks_kde",
    "spatial_pdf_data": "/lustre/fs22/group/icecube/data_mirror/northern_tracks/version-005-p01/KDE_PDFs_v007/IC86_pass2/sig_E_psi_photospline_v006_4D.fits",
}

inj_kwargs = {
    "injection_energy_pdf": injection_energy,
    "injection_sig_time_pdf": injection_time,
    "injection_spatial_pdf": injection_spatial,
}


llh_time = injection_time
llh_spatial = injection_spatial
llh_energy = injection_energy

llh_kwargs = {
    "llh_name": "standard_kde_enabled",
    "llh_energy_pdf": llh_energy,
    "llh_spatial_pdf": llh_spatial,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
    "negative_ns_bool": True,
}

name = "analyses/benchmarks/ps_sens_10yr_NT_KDE+gamma_v2"

# sindecs = np.linspace(0.90, -0.90, 3)
sindecs = np.linspace(0.95, -0.05, 11)
# sindecs = np.linspace(0.5, -0.5, 3)
#
analyses = []

cluster = True

job_ids = []

for sindec in sindecs:
    cat_path = ps_catalogue_name(sindec)

    subname = name + "/sindec=" + "{0:.2f}".format(sindec) + "/"

    scale = (flux_to_k(reference_sensitivity(sindec)) * 2)[0]

    mh_dict = {
        "name": subname,
        "mh_name": "fixed_weights",
        "dataset": nt_v005_p01,
        "catalogue": cat_path,
        "inj_dict": inj_kwargs,
        "llh_dict": llh_kwargs,
        "scale": scale,
        "n_trials": 2000,
        "n_steps": 15,
    }

    if len(sys.argv) > 1 and sys.argv[1] == "--run-jobs":
        job_id = analyse(
            mh_dict,
            cluster=cluster,
            n_cpu=1 if cluster else 16,
            h_cpu="23:59:59",
            ram_per_core="8.0G",
            remove_old_logs=False,
            trials_per_task=50,
        )
        job_ids.append(job_id)

    analyses.append(mh_dict)

if len(sys.argv) > 1 and sys.argv[1] == "--run-jobs":
    wait_cluster(job_ids)


sens = []
sens_err = []
disc_pots = []

for rh_dict in analyses:
    rh = ResultsHandler(rh_dict)
    sens.append(rh.sensitivity)
    sens_err.append(rh.sensitivity_err)
    disc_pots.append(rh.disc_potential)

with open(plot_output_dir(name) + "/PS10yrKDE+gamma.pkl", "wb") as pf:
    pickle.dump(
        {
            "sin_dec": sindecs,
            "sensitivity": list(sens),
            "sens_error": [list(x) for x in list(sens_err)],
            "disc_potential": list(disc_pots),
        },
        pf,
    )

sens_err = np.array(sens_err).T

# sens = reference_sensitivity(sindecs, sample="7yr")
# disc_pots = reference_discovery_potential(sindecs, sample="7yr")
# sens_err = 0.1*sens

plot_range = np.linspace(-0.99, 0.99, 1000)

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
ax1.plot(
    sindecs,
    reference_sensitivity(sindecs, sample="10yr"),
    color="blue",
    label=r"10-year Point Source analysis",
)
# ax1.plot(sindecs, reference_sensitivity(sindecs, sample="7yr"), color="green",
#          label=r"7-year Point Source analysis")

# ax1.plot(sindecs, sens, color='orange', label="Flarestack")

ax1.errorbar(
    sindecs, sens, yerr=sens_err, color="orange", label="Sensitivity", marker="o"
)

ax1.plot(
    sindecs,
    reference_discovery_potential(sindecs, sample="10yr"),
    color="blue",
    linestyle="--",
)

ax1.plot(
    sindecs, disc_pots, color="orange", linestyle="--", label="Discovery Potential"
)

ax1.set_xlim(xmin=-1.0, xmax=1.0)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which="both")
ax1.semilogy(nonpositive="clip")
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]", fontsize=12)

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

sens_ratios = np.array(sens) / reference_sensitivity(sindecs, sample="10yr")
sens_ratio_errs = sens_err / reference_sensitivity(sindecs, sample="10yr")

disc_ratios = np.array(disc_pots) / reference_discovery_potential(
    sindecs, sample="10yr"
)

ax2.errorbar(sindecs, sens_ratios, yerr=sens_ratio_errs, color="red", marker="o")
ax2.scatter(sindecs, disc_ratios, color="k")
ax2.plot(sindecs, disc_ratios, color="k", linestyle="--")
ax2.set_ylabel(r"ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)

plt.suptitle("Point Source Sensitivity (10 year)")
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)

xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

ax1.legend(loc="upper right", fancybox=True, framealpha=1.0)

savefile = plot_output_dir(name) + "/PS10yrKDE+gamma.pdf"

logging.info(f"Saving to {savefile}")

plt.savefig(savefile)
plt.close()

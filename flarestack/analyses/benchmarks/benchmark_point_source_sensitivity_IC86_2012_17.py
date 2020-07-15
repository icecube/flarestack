import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v003_p02
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity,\
    reference_7year_discovery_potential
import matplotlib.pyplot as plt
from flarestack import analyse, wait_cluster
import logging

logging.getLogger().setLevel("INFO")

# Initialise Injectors/LLHs

injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

injection_time = {
    "time_pdf_name": "steady",
}

llh_time = {
    "time_pdf_name": "steady",
}

inj_kwargs = {
    "injection_energy_pdf": injection_energy,
    "injection_sig_time_pdf": injection_time,
}

llh_energy = injection_energy

llh_kwargs = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
}

name = "analyses/benchmarks/ps_sens_ic86"

sindecs = np.linspace(0.90, -0.90, 3)
# sindecs = np.linspace(0.90, -0.90, 9)
# sindecs = np.linspace(0.5, -0.5, 3)

analyses = []

for sindec in sindecs:
    cat_path = ps_catalogue_name(sindec)

    subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    scale = flux_to_k(reference_sensitivity(sindec)) * 6

    mh_dict = {
        "name": subname,
        "mh_name": "fixed_weights",
        "dataset": ps_v003_p02.get_seasons("IC86_2012_17"),
        "catalogue": cat_path,
        "inj_dict": inj_kwargs,
        "llh_dict": llh_kwargs,
        "scale": scale,
        "n_trials": 50,
        "n_steps": 10
    }

    # mh = MinimisationHandler.create(mh_dict)

    analyse(mh_dict, cluster=False, n_cpu=12)

    analyses.append(mh_dict)

wait_cluster()

sens = []
sens_err = []
disc_pots = []

for rh_dict in analyses:
    rh = ResultsHandler(rh_dict)
    sens.append(rh.sensitivity)
    sens_err.append(rh.sensitivity_err)
    disc_pots.append(rh.disc_potential)

sens_err = np.array(sens_err).T

plot_range = np.linspace(-0.99, 0.99, 1000)

plt.figure()
# ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
ax1 = plt.subplot(111)
# ax1.plot(sindecs, reference_sensitivity(sindecs), color="blue",
#          label=r"7-year Point Source analysis")

# ax1.plot(sindecs, sens, color='orange', label="Flarestack")
ax1.errorbar(sindecs, sens, yerr=sens_err, color='orange', label="Sensitivity", marker="o")

# ax1.plot(sindecs, reference_7year_discovery_potential(sindecs), color="blue", linestyle="--")

ax1.plot(
    sindecs, disc_pots, color='orange', linestyle="--", label="Discovery Potential")

ax1.set_xlim(xmin=-1., xmax=1.)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
               fontsize=12)

plt.title('Point Source Sensitivity (1 year)')
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)

xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

# ratio_interp = interp1d(sindecs, sens_ratios)
#
# interp_range = np.linspace(np.min(sindecs),
#                            np.max(sindecs), 1000)

# ax1.plot(
#     interp_range,
#     reference_sensitivity(interp_range)*ratio_interp(interp_range),
#     color='red', linestyle="--", label="Ratio Interpolation")

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

plt.savefig(plot_output_dir(name) + "/1yearPS.pdf")
plt.close()

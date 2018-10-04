import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity,\
    reference_7year_discovery_potential
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

injection_time = {
    "Name": "Steady",
}

llh_time = {
    "Name": "Steady",
}

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

llh_energy = injection_energy

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
}

name = "benchmarks/ps_sens"

sindecs = np.linspace(0.90, -0.90, 13)
# sindecs = np.linspace(0.75, -0.75, 7)
# sindecs = np.linspace(0.5, -0.5, 3)

analyses = []

for sindec in sindecs:
    cat_path = ps_catalogue_name(sindec)

    subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    scale = flux_to_k(reference_sensitivity(sindec)) * 5

    mh_dict = {
        "name": subname,
        "datasets": ps_7year,
        "catalogue": cat_path,
        "inj kwargs": inj_kwargs,
        "llh kwargs": llh_kwargs,
        "scale": scale,
        "n_trials": 5,
        "n_steps": 15
    }

    analysis_path = analysis_dir + subname

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = analysis_path + "dict.pkl"

    with open(pkl_file, "wb") as f:
        Pickle.dump(mh_dict, f)

    # rd.submit_to_cluster(pkl_file, n_jobs=5000)

    # mh = MinimisationHandler(mh_dict)
    # mh.iterate_run(mh_dict["scale"], n_steps=10, n_trials=mh_dict["n_trials"])
    # mh.clear()

    analyses.append(mh_dict)

# rd.wait_for_cluster()

sens = []
disc_pots = []

for rh_dict in analyses:
    rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                        rh_dict["catalogue"])
    sens.append(rh.sensitivity)
    disc_pots.append(rh.disc_potential)

plot_range = np.linspace(-0.99, 0.99, 1000)

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
ax1.plot(sindecs, reference_sensitivity(sindecs), color="blue",
         label=r"7-year Point Source analysis")

ax1.plot(sindecs, sens, color='orange', label="Flarestack")

ax1.plot(sindecs, reference_7year_discovery_potential(sindecs), color="blue", linestyle="--")

ax1.plot(
    sindecs, disc_pots, color='orange', linestyle="--")

ax1.set_xlim(xmin=-1., xmax=1.)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
               fontsize=12)

plt.title('7-year Point Source Sensitivity')

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

sens_ratios = np.array(sens) / reference_sensitivity(sindecs)

disc_ratios = np.array(disc_pots) / reference_7year_discovery_potential(sindecs)

ax2.scatter(sindecs, sens_ratios, color="red")
ax2.plot(sindecs, sens_ratios, color="red")
ax2.scatter(sindecs, disc_ratios, color="k")
ax2.plot(sindecs, disc_ratios, color="k", linestyle="--")
ax2.set_ylabel(r"ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)
# ax2.set_ylim(ymin=0.5, ymax=1.5)
ax2.grid(True)
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

plt.savefig(plot_output_dir(name) + "/7yearPS.pdf")
plt.close()

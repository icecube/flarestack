import numpy as np
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7year import ps_7year
from shared import plot_output_dir, flux_to_k
from utils.prepare_catalogue import ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
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

name = "tests/ps_sens"

sindecs = np.linspace(0.90, -0.90, 13)
sindecs = np.linspace(0.5, -0.5, 3)

sens = []

for sindec in sindecs:
    source_path = ps_catalogue_name(sindec)

    sources = np.load(source_path)

    subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    scale = flux_to_k(skylab_7year_sensitivity(sindec)) * 2

    # mh = MinimisationHandler(subname, ps_7year, sources, inj_kwargs,
    #                          llh_kwargs)
    # mh.iterate_run(scale=scale, n_trials=500, n_steps=10)

    rh = ResultsHandler(subname, llh_kwargs, sources)
    sens.append(rh.sensitivity)

plot_range = np.linspace(-0.99, 0.99, 1000)

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
ax1.plot(plot_range, skylab_7year_sensitivity(plot_range),
         label=r"7-year Point Source analysis")

ax1.scatter(
    sindecs, sens, color='black',
    label='FlareStack')

ax1.set_xlim(xmin=-1., xmax=1.)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{"
               r"-1}$ ]",
               fontsize=12)

plt.title('7-year Point Source Sensitivity')

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

ratios = sens / skylab_7year_sensitivity(sindecs)

ax2.scatter(sindecs, ratios, color="black")
ax2.plot(sindecs, ratios, linestyle="--", color="red")
ax2.set_ylabel(r"ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)
# ax2.set_ylim(ymin=0.5, ymax=1.5)
ax2.grid(True)
xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

ratio_interp = interp1d(sindecs, ratios)

interp_range = np.linspace(np.min(sens),
                           np.max(sens), 1000)

ax1.plot(
    interp_range,
    skylab_7year_sensitivity(interp_range)*ratio_interp(interp_range),
    color='red', linestyle="--", label="Ratio Interpolation")

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

plt.savefig(plot_output_dir(name) + "/7yearPS.pdf")
plt.close()

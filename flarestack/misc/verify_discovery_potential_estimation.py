from __future__ import division
import numpy as np
import os
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.data.icecube.ps_tracks.ps_v003_p02 import ps_10year
from flarestack.data.icecube.gfu.gfu_v002_p04 import gfu_v002_p04
from flarestack.data.icecube.northern_tracks.nt_v002_p05 import diffuse_8year
from flarestack.shared import plot_output_dir, flux_to_k, k_to_flux
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import \
    reference_7year_discovery_potential
import matplotlib.pyplot as plt
from flarestack.analyses.agn_cores.shared_agncores import agn_catalogue_name
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.utils.deus_ex_machina import DeusExMachina

name = "verify_disc_potential"

# Initialise Injectors/LLHs

injection_gamma = 2.0

injection_energy = {
    "Name": "Power Law",
    "Gamma": injection_gamma,
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
    "name": "standard",
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
}

sindecs = np.linspace(0.90, -0.90, 37)
# sindecs = np.linspace(0.75, -0.75, 7)
# sindecs = np.linspace(0.5, -0.5, 3)

# cat_path = agn_catalogue_name("radioloud", "2rxs_100brightest_srcs")
# cat_path = tde_catalogue_name("jetted")

datasets = [
    ("PS Sample (7 year)", ps_7year, True),
    ("PS Sample (10 year)", ps_10year, True),
    ("GFU Sample (8 year)", gfu_v002_p04, True),
    ("Diffuse Sample (8 year)", diffuse_8year, False)
]

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)
ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)
refs = reference_7year_discovery_potential(sindecs, injection_gamma)

ax1.plot(sindecs, refs, label=r"7-year Point Source analysis", color="k")

for j, (sample_name, dataset, fullsky) in enumerate(datasets):

    disc_pots = []

    if fullsky:
        mask = np.ones_like(sindecs, dtype=bool)

    else:
        mask = [sindecs > - 0.05]

    color = ['r', "blue", 'green', 'y'][j]

    dem = DeusExMachina(dataset, inj_kwargs)

    for sindec in sindecs[mask]:
        cat_path = ps_catalogue_name(sindec)

        # scale = flux_to_k(reference_sensitivity(sindec)) * 5
        #
        # mh_dict = {
        #     "name": "",
        #     "mh_name": "fixed_weights",
        #     "datasets": ps_7year,
        #     "catalogue": cat_path,
        #     "inj_dict": inj_kwargs,
        #     "llh_dict": llh_kwargs,
        # }

        # dem.guess_discovery_potential(cat_path)

        # mh = MinimisationHandler.create(mh_dict)
        # print mh.run_trial(0.)
        # raw_input("prompt")

        disc_pots.append(dem.guess_discovery_potential(cat_path))

    disc_pots = k_to_flux(np.array(disc_pots))

    plot_range = np.linspace(-0.99, 0.99, 1000)

    ax1.plot(sindecs[mask], disc_pots, color=color,
             label="Flarestack Estimation ({0})".format(sample_name))

    disc_ratios = np.array(disc_pots)/refs[mask]

    ax2.scatter(sindecs[mask], disc_ratios, color=color)
    ax2.plot(sindecs[mask], disc_ratios, color=color, linestyle="--")

ax1.set_xlim(xmin=-1., xmax=1.)
# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')
ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]",
               fontsize=12)

ax2.set_ylabel(r"ratio", fontsize=12)
ax2.set_xlabel(r"sin($\delta$)", fontsize=12)
#
ax1.set_xlim(xmin=-1.0, xmax=1.0)
# ax2.set_ylim(ymin=0.5, ymax=1.5)
ax2.grid(True)
xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

plt.suptitle('Point Source Discovery Potential')

# ratio_interp = interp1d(sindecs, sens_ratios)
#
# interp_range = np.linspace(np.min(sindecs),
#                            np.max(sindecs), 1000)

# ax1.plot(
#     interp_range,
#     reference_sensitivity(interp_range)*ratio_interp(interp_range),
#     color='red', linestyle="--", label="Ratio Interpolation")

ax1.legend(loc='upper right', fancybox=True, framealpha=1.)

save_dir = plot_output_dir(name)

try:
    os.makedirs(save_dir)
except OSError:
    pass

plt.savefig(save_dir + "/PSDisc.pdf")
plt.close()

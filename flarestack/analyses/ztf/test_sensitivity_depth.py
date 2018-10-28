import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
import datetime
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.injector import Injector
import matplotlib.pyplot as plt
import math
from flarestack.utils.prepare_catalogue import cat_dtype
from flarestack.utils.neutrino_cosmology import ccsn_madau, \
    ccsn_clash_candels, define_cosmology_functions, integrate_over_z, \
    cumulative_z
from scipy.interpolate import interp1d
from flarestack.analyses.ztf.simulate_ccsn_catalogue import cat_names, \
    inj_kwargs, pre_window

name_root = "analyses/ztf/depth/"


# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

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

res_dict = dict()


for i, cat in enumerate(cat_names):

    name = name_root + os.path.basename(cat)[:-4] + "/"

    # cat_path = catalogue_dir + "TDEs/TDE_silver_catalogue.npy"
    # catalogue = np.load(cat_path)

    scale = flux_to_k(reference_sensitivity(
        np.sin(0.0))) * 160

    mh_dict = {
        "name": name,
        "datasets": [IC86_1_dict],
        "catalogue": cat,
        "inj kwargs": inj_kwargs,
        "llh kwargs": llh_kwargs,
        "scale": scale,
        "n_trials": 5,
        "n_steps": 30
    }

    analysis_path = analysis_dir + name

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = analysis_path + "dict.pkl"

    with open(pkl_file, "wb") as f:
        Pickle.dump(mh_dict, f)

    start = datetime.datetime.now()

    # mh = MinimisationHandler(mh_dict)
    # mh.iterate_run(scale=scale, n_steps=mh_dict["n_steps"],
    #                n_trials=mh_dict["n_trials"])
    # mh.clear()
    rd.submit_to_cluster(pkl_file, n_jobs=100)

    res_dict[float(i)] = mh_dict

rd.wait_for_cluster()

sens = []
sens_e = []
disc = []
disc_e = []

dist = []

print res_dict.keys()

for (n_cat, rh_dict) in sorted(res_dict.iteritems()):
    rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                        rh_dict["catalogue"], show_inj=True)

    inj_time = pre_window * 60 * 60 * 24

    astro_sens, astro_disc = rh.astro_values(
        rh_dict["inj kwargs"]["Injection Energy PDF"])

    key = "Total Fluence (GeV cm^{-2} s^{-1})"

    e_key = "Mean Luminosity (erg/s)"

    sens.append(astro_sens[key] * inj_time)
    disc.append(astro_disc[key] * inj_time)

    sens_e.append(astro_sens[e_key] * inj_time)
    disc_e.append(astro_disc[e_key] * inj_time)

    cat = np.load(rh_dict["catalogue"])
    dist.append(max(cat["Distance (Mpc)"]))

savedir = plot_output_dir(name_root)

try:
    os.makedirs(os.path.dirname(savedir))
except OSError:
    pass

plt.figure()
ax1 = plt.subplot(111)
# ax2 = ax1.twinx()
ax1.plot(dist, sens)
# ax1.plot(dist, disc)
plt.xlabel("Distance (Mpc)")
ax1.set_ylabel(r"Time-Integrated Flux [ GeV$^{-1}$ cm$^{-2}$]")
plt.savefig(savedir + "detected_flux.pdf")
plt.close()

plt.figure()
ax1 = plt.subplot(111)
ax1.plot(dist, sens_e)
# ax1.plot(dist, disc_e)
plt.xlabel("Distance (Mpc)")
ax1.set_ylabel(r"Energy per source (erg)")
plt.savefig(savedir + "e_per_source.pdf")
plt.close()


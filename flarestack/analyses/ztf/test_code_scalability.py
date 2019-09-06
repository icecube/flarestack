from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import numpy as np
import datetime
import os
import pickle as Pickle
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.core.minimisation import MinimisationHandler
import matplotlib.pyplot as plt
import math
from flarestack.utils.prepare_catalogue import cat_dtype

name_root = "analyses/ztf/scalability/"

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

# Use a source that is constant in time

injection_time = {
    "Name": "Steady",
    "Pre-Window": 5,
    "Post-Window": 0
}

# Use a source with a spectral index of -2, with an energy range between
# 100 GeV and 10 Pev (10**7 GeV).

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
    "E Min": 10**2,
    "E Max": 10**7
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Look for a source that is constant in time

llh_time = dict(injection_time)

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

n_catalogue = np.logspace(0, 5, 6)
n_catalogue = np.logspace(0, 4, 17)

print("Entries in catalogue", n_catalogue)

times = []

for n in n_catalogue:

    name = name_root + str(n) + "/"

    catalogue = np.empty(int(n), dtype=cat_dtype)

    catalogue["Name"] = ["src" + str(i) for i in range(int(n))]
    catalogue["ra"] = np.random.uniform(0., 2*np.pi, int(n))
    catalogue["dec"] = np.random.uniform(-0.5 * np.pi, 0.5*np.pi, int(n))
    catalogue["Relative Injection Weight"] = np.exp(
        np.random.normal(0., 2., int(n))
    )
    catalogue["Distance (Mpc)"] = np.ones(int(n))
    catalogue["Ref Time (MJD)"] = np.random.uniform(55710., 56010, int(n))

    cat_path = catalogue_dir + "random/" + str(n) + "_cat.npy"
    try:
        os.makedirs(os.path.dirname(cat_path))
    except OSError:
        pass
    np.save(cat_path, catalogue)

    # cat_path = catalogue_dir + "TDEs/TDE_silver_catalogue.npy"
    # catalogue = np.load(cat_path)

    scale = flux_to_k(reference_sensitivity(
        np.sin(0.0))) * 40 * math.sqrt(float(n))

    mh_dict = {
        "name": name,
        "datasets": ps_v002_p01,
        "catalogue": cat_path,
        "inj kwargs": inj_kwargs,
        "llh kwargs": llh_kwargs,
        "scale": scale,
        "n_trials": 5,
        "n_steps": 15
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

    n_trials = 10

    mh = MinimisationHandler(mh_dict)
    mh.run(n_trials=n_trials, scale=0.0)
    mh.clear()
    end = datetime.datetime.now()
    diff = (end-start).total_seconds() / n_trials

    times.append(diff)

savepath = plot_output_dir(name_root) + "scale.pdf"

try:
    os.makedirs(os.path.dirname(savepath))
except OSError:
    pass

plt.figure()
plt.plot(n_catalogue, times)
plt.xlabel("Entries in Catalogue")
plt.ylabel("Length of 1 background scramble (s)")
plt.xscale("log")
plt.yscale("log")
plt.savefig(savepath)
plt.close()


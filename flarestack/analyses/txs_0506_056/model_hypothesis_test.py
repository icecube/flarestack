from __future__ import print_function
from builtins import range
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
import random
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict,\
    IC86_234_dict
from flarestack.shared import flux_to_k, make_analysis_pickle, plot_output_dir
from flarestack.core.minimisation import MinimisationHandler
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.analyses.txs_0506_056.make_txs_catalogue import \
    txs_catalogue, txs_cat_path
from flarestack.analyses.txs_0506_056.load_gao_spectral_models import \
    spline_name
from flarestack.utils.custom_dataset import custom_dataset
from scipy.stats import norm

base_dir = "analyses/txs_0506_056/model_hypothesis_test/"

# Initialise Injectors/LLHs

inj_dict = {
    "Injection Time PDF": {
        "Name": "FixedEndBox"
    },
    "Injection Energy PDF": {
        "Name": "Power Law",
        "Gamma": 2.18,
    },
    "fixed_n": 13
}

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.


# Look for a source that is constant in time

llh_time = {
    "Name": "FixedEndBox"
}

# Try to fit a power law to the data

llh_energy_pl = {
    "Name": "Power Law"
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs_pl = {
    "name": "standard",
    "LLH Energy PDF": llh_energy_pl,
    "LLH Time PDF": llh_time,
}


# Creates the Minimisation Handler dictionary, which contains all relevant
# information to run an analysis

mh_dict_pl = {
    "name": base_dir + "power_law/",
    "mh_name": "fixed_weights",
    "datasets": [IC86_234_dict],
    "catalogue": txs_cat_path,
    "inj kwargs": inj_dict,
    "llh_dict": llh_kwargs_pl
}

# Try to fit a power law to the data


llh_energy_tm = {
        "Name": "Spline",
        "Spline Path": spline_name(0)
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs_tm = {
    "name": "fixed_energy",
    "LLH Energy PDF": llh_energy_tm,
    "LLH Time PDF": llh_time,
}


# Creates the Minimisation Handler dictionary, which contains all relevant
# information to run an analysis

mh_dict_tm = {
    "name": base_dir + "theory_model/",
    "mh_name": "fixed_weights",
    "datasets": [IC86_234_dict],
    "catalogue": txs_cat_path,
    "inj kwargs": inj_dict,
    "llh_dict": llh_kwargs_tm
}

ts_path = plot_output_dir(base_dir) + "model_TS.pkl"

print("TS path", ts_path)

try:
    os.makedirs(os.path.dirname(ts_path))
except OSError:
    pass


if os.path.isfile(ts_path):
    with open(ts_path, "r") as f:
        print("Loading ts_array")
        ts_array = Pickle.load(f)

else:
    print("Empty TS array")
    ts_array = []

# Creates a Minimisation Handler using the dictionary, and runs the trials

mh_pl = MinimisationHandler.create(mh_dict_pl)
mh_tm = MinimisationHandler.create(mh_dict_tm)

n_trials = 100

for i in range(n_trials):

    seed = random.randint(0, 999999)
    mh_pl.set_random_seed(seed)
    res_pl = mh_pl.run_trial(scale=1.)
    mh_tm.set_random_seed(seed)
    res_tm = mh_tm.run_trial(scale=1.)
    ts = res_tm["TS"] - res_pl["TS"]
    print(i, seed, res_tm, res_pl, ts)
    ts_array.append(ts)


with open(ts_path, "wb") as f:
    Pickle.dump(ts_array, f)

weights = np.ones_like(ts_array)
weights /= np.sum(weights)

print(len(ts_array), "trials")

savepath = plot_output_dir(base_dir) + "TS.pdf"
plt.figure()

result_ts = 2.7

plt.hist(ts_array, bins=50, lw=2, histtype='step', weights=weights)
plt.axvline(result_ts, color="orange", linestyle=":")


plt.yscale("log")
plt.xlabel(r"Test Statistic ($\lambda$)")
# plt.legend(loc="upper right")

try:
    os.makedirs(os.path.dirname(savepath))
except OSError:
    pass

print("Saving to", savepath)
plt.savefig(savepath)
plt.close()

ts_array = np.array(ts_array)

n_over = np.sum([ts_array > result_ts])
if n_over == 0:
    print("No trials above tested value. More statistics needed. We will " \
          "assume that 1 was found, to give a conservative limit.")
    n_over = 1.
pvalue = n_over/float(len(ts_array))

print("P-value:", pvalue)

print("Sigma:", norm.ppf(1-pvalue))






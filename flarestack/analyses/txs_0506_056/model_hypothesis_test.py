import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
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
from flarestack.utils.custom_seasons import custom_dataset

base_dir = "analyses/txs_0506_056/model_hypothesis_test/"

# Initialise Injectors/LLHs

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
    "inj kwargs": {},
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
    "inj kwargs": {},
    "llh_dict": llh_kwargs_tm
}

# Creates a Minimisation Handler using the dictionary, and runs the trials

mh_pl = MinimisationHandler.create(mh_dict_pl)
mh_tm = MinimisationHandler.create(mh_dict_tm)

seed = 45

mh_pl.set_random_seed(seed)
res_pl = mh_pl.run_trial(scale=0.)["TS"]
mh_tm.set_random_seed(seed)
res_tm = mh_tm.run_trial(scale=0.)["TS"]
ts = res_tm - res_pl
print res_tm, res_pl, ts





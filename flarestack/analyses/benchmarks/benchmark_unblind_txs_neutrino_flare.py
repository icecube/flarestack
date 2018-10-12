"""Script to replicate unblinding of the neutrino flare found for the source
TXS 0506+56, as described in https://arxiv.org/abs/1807.08794.
"""
import numpy as np
import os
import cPickle as Pickle
from flarestack.core.unblinding import Unblinder
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
from flarestack.analyses.txs_0506_056.make_txs_catalogue import txs_cat_path,\
    txs_catalogue
from flarestack.utils.custom_seasons import custom_dataset

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "Name": "Power Law",
}

llh_time = {
    "Name": "FixedEndBox"
}

unblind_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
}

name = "analyses/benchmarks/TXS_0506+056/"

unblind_dict = {
    "name": name,
    "datasets": custom_dataset(txs_sample_v1, txs_catalogue,
                               unblind_llh["LLH Time PDF"]),
    "catalogue": txs_cat_path,
    "llh kwargs": unblind_llh,
}

ub = Unblinder(unblind_dict, mock_unblind=False)

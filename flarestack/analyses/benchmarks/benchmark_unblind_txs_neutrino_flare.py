"""Script to replicate unblinding of the neutrino flare found for the source
TXS 0506+56, as described in https://arxiv.org/abs/1807.08794.
"""
import numpy as np
import os
import cPickle as Pickle
from flarestack.core.unblinding import Unblinder
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_234_dict
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
llh_time = {
    "Name": "Steady"
}

# [
#     13.25549565470768,
#     2.1484376675443446,
#     56937.818975052855,
#     57089.43956528731,
#     145.30113144338247,
#     29.48696662
# ]
#
# llh_time = {
#     "Name": "FixedRefBox",
#     "Fixed Ref Time (MJD)": 56937.81,
#     "Pre-Window": 0,
#     "Post-Window": 57089.5 - 56937.81
#     # "Max Flare": 300
# }

unblind_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Flare Search?": True
}

name = "analyses/benchmarks/TXS_0506+056/"

unblind_dict = {
    "name": name,
    # "datasets": custom_dataset(txs_sample_v1, txs_catalogue,
    #                            unblind_llh["LLH Time PDF"]),
    "datasets": [IC86_234_dict],
    "catalogue": txs_cat_path,
    "llh kwargs": unblind_llh,
}

ub = Unblinder(unblind_dict, mock_unblind=False)

import numpy as np
import os
import cPickle as Pickle
from flarestack.core.unblinding import Unblinder
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir, \
    catalogue_dir
import matplotlib.pyplot as plt
from flarestack.utils.custom_seasons import custom_dataset

analyses = dict()

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

unblind_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Fit Weights?": True
}

name_root = "analyses/tde/unblind_stacked_TDEs/"
bkg_ts_root = "analyses/tde/compare_spectral_indices/Emin=100/"

cat_res = dict()

cats = [
    "jetted",
    "gold",
    "obscured",
    "silver"
]

for j, cat in enumerate(cats):

    name = name_root + cat.replace(" ", "") + "/"

    bkg_ts = bkg_ts_root + cat.replace(" ", "") + "/fit_weights/"

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    unblind_dict = {
        "name": name,
        "datasets": custom_dataset(txs_sample_v1, catalogue,
                                   unblind_llh["LLH Time PDF"]),
        "catalogue": cat_path,
        "llh kwargs": unblind_llh,
        "background TS": bkg_ts
    }

    ub = Unblinder(unblind_dict)
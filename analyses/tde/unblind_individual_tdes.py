import numpy as np
import os
import cPickle as Pickle
from core.unblinding import Unblinder
from core.results import ResultsHandler
from data.icecube_gfu_v002_p01 import txs_sample_v1
from shared import plot_output_dir, flux_to_k, analysis_dir, catalogue_dir
import matplotlib.pyplot as plt
from utils.custom_seasons import custom_dataset

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
    "Flare Search?": True
}

name_root = "analyses/tde/unblind_individual_tdes/"
bkg_ts_root = "analyses/tde/compare_spectral_indices_individual/"

cat_res = dict()

cats = [
    "Swift J1644+57",
    "Swift J2058+05",
    "ASASSN-14li",
    "XMMSL1 J0740-85"
]

for j, cat in enumerate(cats):

    name = name_root + cat.replace(" ", "") + "/"

    bkg_ts = bkg_ts_root + cat.replace(" ", "") + "/negative_n_s/2.0/"

    cat_path = catalogue_dir + "TDEs/individual_TDEs/" + cat + "_catalogue.npy"
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

    # rh = ResultsHandler(unblind_dict["name"], unblind_dict["llh kwargs"],
    #                     unblind_dict["catalogue"])
    #
    # astro_sens, astro_disc = rh.astro_values(
    #     unblind_dict["inj kwargs"]["Injection Energy PDF"])
    #
    # key = "Total Fluence (GeV^{-1} cm^{-2} s^{-1})"
    #
    # e_key = "Mean Luminosity (erg/s)"
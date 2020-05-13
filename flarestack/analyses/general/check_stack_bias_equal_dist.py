import numpy as np
import os
from astropy.table import Table
import argparse
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits, get_figure_limits, p_vals
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import sn_cats, updated_sn_catalogue_name, \
    sn_time_pdfs, raw_output_dir, pdf_names, limit_sens
from flarestack.analyses.ccsn import get_sn_color
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
import math
import pickle
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging
import time
logging.getLogger().setLevel("INFO")
injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0
}

injection_time = {
    'time_pdf_name': 'steady'
}

llh_energy = {
    "energy_pdf_name": "power_law"
}

llh_time = {
    'time_pdf_name': 'steady'
}

inj_dict = {
    'injection_energy_pdf': injection_energy,
    'injection_sig_time_pdf': injection_time
}

llh_dict = {
    "llh_name": "standard_matrix",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
}


catalogue = np.load(updated_sn_catalogue_name('IIn'))
catalogue['distance_mpc'] = np.array([1] * len(catalogue))
dir_path = os.path.dirname(os.path.realpath(__file__))
temp_save_catalogue_to = f'{dir_path}/temp_check_stack_bias_equal_dist_cat.npy'
np.save(temp_save_catalogue_to, catalogue)


mh_dict = {
    "name": "examples/crosscheck_stacking_equal_dist/",
    "mh_name": 'large_catalogue',
    "dataset": ps_v002_p01.get_seasons(),
    "catalogue": temp_save_catalogue_to,
#     "catalogue": ps_stack_catalogue_name(0.1, 0.3),
#     "catalogue": tde_catalogue_name("jetted"),
    "inj_dict": inj_dict,
    "llh_dict": llh_dict,
    "scale": 10.,
    "n_trials": 30,
    "n_steps": 10
}

analyse(mh_dict, cluster=False, n_cpu=32)
rh = ResultsHandler(mh_dict)

os.remove(temp_save_catalogue_to)
import unittest
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.data.icecube.gfu.gfu_v002_p01  import gfu_v002_p01
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.minimisation import MinimisationHandler
from flarestack.shared import fs_scratch_dir, flux_to_k, make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.core.results import ResultsHandler
from flarestack.analyses.agn_cores.shared_agncores import agn_catalogue_name

from flarestack.cluster import run_desy_cluster as rd

# Initialise Injectors/LLHs

gamma = 2.0  # gamma of injection   # above 3 this is not enough (to few injected source, you have to change it manually)

name = "analyses/agn_cores/test_analysis/"


llh_dict = {
    "name": "standard_overlapping",
    "LLH Time PDF": {
        "Name": "Steady"
        },
    "LLH Energy PDF":  {
        "Name": "Power Law"
        }
    }

# llh_dict = {
#     "name": "spatial",
#     "LLH Time PDF": {
#         "Name": "Steady"
#         },
#     }

inj_dict = {
    "Injection Time PDF": {
        "Name": "Steady"
        },
    "Injection Energy PDF": {
        "Name": "Power Law",
        "Gamma": gamma,
        }
    }

mh_dict = {
    "name": name,
    "mh_name": "fixed_weights",
    "datasets": ps_7year[-2:-1],
    "catalogue":agn_catalogue_name("radioloud", "2rxs_100brightest_srcs_dec0_weight1"),
       # agn_catalogue_name("random", "NorthSky_2close_srcs"),   # two close sources
       # agn_catalogue_name("random", "NorthSky_100brightest_srcs_dec0_weight1"), # 100random sources equally separated with same weight1
       # agn_catalogue_name("radioloud", "2rxs_100brightest_srcs_dec0_weight1"),
       # agn_catalogue_name("random", "2rxs_100brightest_srcs_weight1"),
       # agn_catalogue_name("random", "2rxs_100brightest_srcs"),
       # agn_catalogue_name("radioloud", "2rxs_100random_srcs"),
       # agn_catalogue_name("radioloud", "2rxs_test"),
    "llh_dict": llh_dict,
    "inj kwargs": inj_dict,
    "n_trials": 50,
    "n_steps": 10
    }

# cat_name =  agn_catalogue_name("random", "2rxs_100brightest_srcs")
# cat_name =  agn_catalogue_name("radioloud", "2rxs_100brightest_srcs_dec0_weight1")
# cat = np.load(cat_name)
# print "Cat is ", cat_name, " Its lenght is: ", len(cat)

scale = flux_to_k(reference_sensitivity(0.5, gamma)*20) #*20*10**-3   #0.5 is the usally the sin_dec of the closest source  -> [this produced 60000 neutrinos!!!
mh_dict["scale"] = scale
pkl_file = make_analysis_pickle(mh_dict)

# rd.submit_to_cluster(pkl_file, n_jobs=20)
rd.wait_for_cluster()
# mh = MinimisationHandler.create(mh_dict)

# mh.iterate_run(scale=scale, n_steps=10, n_trials=50)
# mh.iterate_run(scale=scale, n_steps=3, n_trials=5)
rh = ResultsHandler(mh_dict)

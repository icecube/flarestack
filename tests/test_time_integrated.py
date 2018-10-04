"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1). The sensitivity is calculated for three declinations,
and compared to stored values for these results. If the sensitivity deviates
from the stored sensitivity by more than X%, the test is failed.

The approximate runtime for this test, in which 1000 background trials are
run, is roughly 1 minute.

"""
import unittest
import numpy as np
import os
import datetime
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity
import flarestack.cluster.run_desy_cluster as rd
from flarestack.core.minimisation import MinimisationHandler

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

injection_time = {
    "Name": "Steady",
}

llh_time = {
    "Name": "Steady",
}

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

llh_energy = injection_energy

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
}

# Generate a dynamic name to ensure pickles from previous test runs cannot
# contaminate results

name = "tests/time_integrated_" + datetime.datetime.now().strftime(
    "%Y-%m-%d_%H:%M:%S")

# Loop over sin(dec) values

sindecs = np.linspace(0.5, -0.5, 3)


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

high_statistics_results = [
    2.097237550036032e-09,
    1.4530108421250116e-09,
    1.6870891536361605e-08
]


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):
        pass


all_sens = []

for i, sindec in enumerate(sindecs):
    cat_path = ps_catalogue_name(sindec)

    subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    scale = flux_to_k(reference_sensitivity(sindec)) * 7

    mh_dict = {
        "name": subname,
        "datasets": [IC86_1_dict],
        "catalogue": cat_path,
        "inj kwargs": inj_kwargs,
        "llh kwargs": llh_kwargs,
        "scale": scale,
        "n_trials": 100,
        "n_steps": 10
    }

    analysis_path = analysis_dir + subname

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = analysis_path + "dict.pkl"

    with open(pkl_file, "wb") as f:
        Pickle.dump(mh_dict, f)

    # mh = MinimisationHandler(mh_dict)
    # mh.iterate_run(mh_dict["scale"], n_steps=mh_dict["n_steps"],
    #                n_trials=mh_dict["n_trials"])
    # mh.clear()
    rd.submit_to_cluster(pkl_file, n_jobs=100)
    rd.wait_for_cluster()

    rh = ResultsHandler(mh_dict["name"], mh_dict["llh kwargs"],
                        mh_dict["catalogue"])

    sens = rh.sensitivity
    ref = high_statistics_results[i]

    print sens

    print ref

    frac_diff = abs(sens-ref)/ref

    print "Fractional deviation", frac_diff

    all_sens.append(sens)

print all_sens

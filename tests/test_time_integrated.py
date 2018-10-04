"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1). The sensitivity is calculated for three declinations,
and compared to stored values for these results. If the sensitivity deviates
from the stored sensitivity by more than 20%, the test is failed.

The approximate runtime for this test, in which 2000 background trials are
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
    2.262657165737197e-09,
    1.4715209683167835e-09,
    1.5604438427616364e-08
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        # Test threev= declinations

        for i, sindec in enumerate(sindecs):
            cat_path = ps_catalogue_name(sindec)
            subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

            # Set characteristic flux scale
            scale = flux_to_k(reference_sensitivity(sindec)) * 7

            mh_dict = {
                "name": subname,
                "datasets": [IC86_1_dict],
                "catalogue": cat_path,
                "inj kwargs": inj_kwargs,
                "llh kwargs": llh_kwargs,
                "scale": scale,
                "n_trials": 200,
                "n_steps": 10
            }

            # Run trials

            mh = MinimisationHandler(mh_dict)
            mh.iterate_run(mh_dict["scale"], n_steps=mh_dict["n_steps"],
                           n_trials=mh_dict["n_trials"])
            mh.clear()

            # Calculate sensitivity

            rh = ResultsHandler(mh_dict["name"], mh_dict["llh kwargs"],
                                mh_dict["catalogue"])

            sens = rh.sensitivity
            ref = high_statistics_results[i]

            frac_diff = abs(sens-ref)/ref

            print "\n"
            print "\n"
            print "Fractional Deviation:", frac_diff
            print "\n"
            print "\n"

            self.assertLess(frac_diff, 0.2)


if __name__ == '__main__':
    unittest.main()

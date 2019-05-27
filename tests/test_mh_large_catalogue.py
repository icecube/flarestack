"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
from __future__ import print_function
import unittest
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.unblinding import create_unblinder
from flarestack.core.minimisation import MinimisationHandler
from flarestack.analyses.agn_cores.shared_agncores import agn_subset_catalogue

# Initialise Injectors/LLHs

llh_dict = {
    "name": "standard_matrix",
    "llh_time_pdf": {
        "time_pdf_name": "Steady"
    },
    "llh_energy_pdf": {
        "energy_pdf_name": "PowerLaw"
    }
}

inj_dict = {
    "injection_time_pdf": {
        "time_pdf_name": "Steady"
    },
    "injection_energy_pdf": {
        "energy_pdf_name": "PowerLaw",
        "gamma": 2.0
    }
}

# Create a catalogue containing the 700 brightest sources in the radioloud
# AGN core analysis. This will  be used with IC40 to stress-test the
# 'large_catalogue method for many sources.

n_sources = 150

catalogue = agn_subset_catalogue("radioloud", "radioselected", n_sources)


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [
    [0.0, 2.7849363898525303],
    [4.899457256050394, 1.9038645526134297]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print("\n")
        print("\n")
        print("Testing 'large_catalogue' MinimisationHandler class "
              "with {0} sources and IC40 data".format(n_sources))
        print("\n")
        print("\n")

        # Test stacking

        unblind_dict = {
            "mh_name": "large_catalogue",
            "datasets": ps_7year.get_seasons("IC40"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
            "inj_dict": {}
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters[0])

        print("Best fit values", list(res["x"]))
        print("Reference best fit", true_parameters[0])

        mh_dict = {
            "name": "tests/test_mh_large_catalogue",
            "mh_name": "large_catalogue",
            "datasets": ps_7year.get_seasons("IC40"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
            "inj_dict": inj_dict,
            "scale": 10,
            "n_steps": 5,
            "n_trials": 5
        }

        mh = MinimisationHandler.create(mh_dict)
        res = mh.simulate_and_run(10)["res"]
        print(res)
        self.assertEqual(list(res["x"]), true_parameters[1])

        print("Best fit values", list(res["x"]))
        print("Reference best fit", true_parameters[1])


if __name__ == '__main__':
    unittest.main()

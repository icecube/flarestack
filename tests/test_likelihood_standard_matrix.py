"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
from __future__ import print_function
import unittest
from flarestack.data.icecube import ps_v002_p01
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name

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

name = "tests/test_likelihood_spatial/"

# Loop over sin(dec) values

catalogue = tde_catalogue_name("jetted")


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [2.3300319373809333, 1.8385540231080426]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print("\n")
        print("\n")
        print("Testing 'standard_matrix' LLH class")
        print("\n")
        print("\n")

        # Test stacking

        unblind_dict = {
            "mh_name": "fixed_weights",
            "datasets": ps_v002_p01.get_seasons("IC86_1"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters)

        print("Best fit values", list(res["x"]))
        print("Reference best fit", true_parameters)


if __name__ == '__main__':
    unittest.main()

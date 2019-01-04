"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import unittest
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.utils.custom_seasons import custom_dataset
import numpy as np

# Initialise Injectors/LLHs

llh_dict = {
    "name": "standard",
    "LLH Time PDF": {
        "Name": "FixedEndBox"
    },
    "LLH Energy PDF": {
        "Name": "Power Law"
    }
}

name = "tests/test_mh_fit_weights/"

true_parameters = [2.3834973689924817, 0.0, 0.0, 2.12991507453312]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print "\n"
        print "\n"
        print "Testing fit_weight MinimisationHandler class"
        print "\n"
        print "\n"

        mh_name = "fit_weights"

        # Test three declinations

        unblind_dict = {
            "name": name,
            "mh_name": mh_name,
            "datasets": custom_dataset(ps_7year, np.load(catalogue),
                                       llh_dict["LLH Time PDF"]),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
            "llh kwargs": llh_dict
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.iterkeys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters)

        print "Best fit values", list(res["x"])
        print "Reference best fit", true_parameters


if __name__ == '__main__':
    unittest.main()

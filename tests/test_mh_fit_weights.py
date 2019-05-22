"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
from __future__ import print_function
import unittest
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.utils.custom_seasons import custom_dataset
from flarestack.utils.catalogue_loader import load_catalogue
import numpy as np

# Initialise Injectors/LLHs

llh_dict = {
    "llh_name": "standard",
    "llh_time_pdf": {
        "time_pdf_name": "FixedEndBox"
    },
    "llh_energy_pdf": {
        "energy_pdf_name": "Power Law"
    }
}

true_parameters = [
    2.2125203223350054, 0.0, 0.186667537993545, 2.1249999977127216
]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print("\n")
        print("\n")
        print("Testing 'fit_weight' MinimisationHandler class")
        print("\n")
        print("\n")

        mh_name = "fit_weights"

        # Test three declinations

        unblind_dict = {
            "mh_name": mh_name,
            "datasets": custom_dataset(ps_7year, load_catalogue(catalogue),
                                       llh_dict["llh_time_pdf"]),
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

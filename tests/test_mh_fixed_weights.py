"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.icecube import ps_v002_p01
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name

# Initialise Injectors/LLHs

llh_dict = {
    "llh_name": "standard",
    "llh_sig_time_pdf": {
        "time_pdf_name": "steady"
    },
    "llh_bkg_time_pdf": {
        "time_pdf_name": "steady",
    },
    "llh_energy_pdf": {
        "energy_pdf_name": "power_law"
    }
}

true_parameters = [2.4806329060480596, 1.850001327552612]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'fixed_weight' MinimisationHandler class")

        # Test three declinations

        unblind_dict = {
            "mh_name": "fixed_weights",
            "dataset": ps_v002_p01.get_seasons("IC86_1"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters)

        logging.info("Best fit values {0}".format(list(res)))
        logging.info("Reference best fit {0}".format(true_parameters))


if __name__ == '__main__':
    unittest.main()

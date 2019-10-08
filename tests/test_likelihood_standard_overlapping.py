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
    "llh_name": "standard_overlapping",
    "llh_sig_time_pdf": {
        "time_pdf_name": "steady"
    },
    "llh_bkg_time_pdf": {
        "time_pdf_name": "steady"
    },
    "llh_energy_pdf": {
        "energy_pdf_name": "power_law"
    }
}

# Loop over sin(dec) values

catalogue = tde_catalogue_name("jetted")


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [2.3300332566811863, 1.8385541332476598]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'standard_overlapping' LLH class")

        # Test stacking

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

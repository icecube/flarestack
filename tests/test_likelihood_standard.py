"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
import numpy as np
from flarestack.data.icecube import ps_v002_p01
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.unblinding import create_unblinder

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

# Loop over sin(dec) values

sindecs = np.linspace(0.5, -0.5, 3)


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [
    [2.8524121731705296, 4.0],
    [0.0, 3.012888497161147],
    [2.611394680431712, 2.7978753815656057]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'standard' LLH class")

        # Test three declinations

        for j, sindec in enumerate(sindecs):

            unblind_dict = {
                "mh_name": "fixed_weights",
                "dataset": ps_v002_p01.get_seasons("IC86_1"),
                "catalogue": ps_catalogue_name(sindec),
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]
            self.assertEqual(list(res["x"]), true_parameters[j])

            logging.info("Best fit values {0}".format(list(res)))
            logging.info("Reference best fit {0}".format(true_parameters[j]))


if __name__ == '__main__':
    unittest.main()

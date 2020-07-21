"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.unblinding import create_unblinder
from flarestack.core.minimisation import MinimisationHandler
from flarestack.analyses.agn_cores.shared_agncores import agn_subset_catalogue

# Initialise Injectors/LLHs

llh_dict = {
    "llh_name": "standard_matrix",
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

inj_dict = {
    "injection_sig_time_pdf": {
        "time_pdf_name": "steady"
    },
    "injection_energy_pdf": {
        "energy_pdf_name": "power_law",
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
    [0.0, 2.33905480645302],
    [14.379477037814556, 4.0]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'large_catalogue' MinimisationHandler class "
              "with {0} sources and IC40 data".format(n_sources))

        # Test stacking

        unblind_dict = {
            "name": "test/test_large_catalogue/",
            "mh_name": "large_catalogue",
            "dataset": icecube_ps_3_year.get_seasons("IC79-2010"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
            "inj_dict": {}
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]

        logging.info("Best fit values {0}".format(list(res["x"])))
        logging.info("Reference best fit {0}".format(true_parameters[0]))

        for i, x in enumerate(res["x"]):
            self.assertAlmostEqual(x, true_parameters[0][i], delta=0.1)

        # mh_dict = {
        #     "name": "tests/test_mh_large_catalogue",
        #     "mh_name": "large_catalogue",
        #     "dataset": icecube_ps_3_year.get_seasons("IC79-2010"),
        #     "catalogue": catalogue,
        #     "llh_dict": llh_dict,
        #     "inj_dict": inj_dict,
        #     "scale": 10.,
        #     "n_steps": 5,
        #     "n_trials": 5
        # }
        #
        # mh = MinimisationHandler.create(mh_dict)
        # res = mh.simulate_and_run(10)["res"]
        # self.assertEqual(list(res["x"]), true_parameters[1])
        #
        # logging.info("Best fit values {0}".format(list(res)))
        # logging.info("Reference best fit {0}".format(true_parameters[1]))


if __name__ == '__main__':
    unittest.main()

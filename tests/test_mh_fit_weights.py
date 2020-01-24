"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack import MinimisationHandler, analyse

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

true_parameters = [
    3.6400763376308523, 0.0, 0.0, 4.0
]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'fit_weight' MinimisationHandler class")

        mh_name = "fit_weights"

        # Test three declinations

        unblind_dict = {
            "name": "tests/test_mh_fit_weights",
            "mh_name": mh_name,
            "dataset": icecube_ps_3_year.get_seasons("IC86-2011"),
            "catalogue": catalogue,
            "llh_dict": llh_dict,
        }

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        for i, x in enumerate(res["x"]):
            self.assertAlmostEqual(x, true_parameters[i], delta=0.1)
        logging.info("Best fit values {0}".format(list(res)))
        logging.info("Reference best fit {0}".format(true_parameters))

        inj_dict = {
            "injection_sig_time_pdf": {
                "time_pdf_name": "steady"
            },
            "injection_bkg_time_pdf": {
                "time_pdf_name": "steady",
            },
            "injection_energy_pdf": {
                "energy_pdf_name": "power_law",
                "gamma": 2.0
            }
        }

        mh_dict = dict(unblind_dict)
        mh_dict["inj_dict"] = inj_dict
        mh_dict["n_trials"] = 1.
        mh_dict["n_steps"] = 3.
        mh_dict["scale"] = 5.

        mh = MinimisationHandler.create(mh_dict)
        res = mh.simulate_and_run(5.)
        analyse(mh_dict, cluster=False)

if __name__ == '__main__':
    unittest.main()

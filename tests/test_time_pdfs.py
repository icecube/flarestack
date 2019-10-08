"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.icecube import ps_v002_p01
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name

# Initialise Injectors/LLHs

time_pdfs = [
    {
        "time_pdf_name": "steady"
    },
    {
        "time_pdf_name": "box",
        "pre_window": 0.,
        "post_window": 100.
    },
    {
        "time_pdf_name": "custom_source_box",
    },
    {
        "time_pdf_name": "fixed_ref_box",
        "pre_window": 0.,
        "post_window": 100.,
        "fixed_ref_time_mjd": 56000
    }
]

true_parameters = [
    [8.302537737393378, 2.0749999999312245],
    [2.6097792634284898, 2.099999997928826],
    [2.994109963445442, 2.099999998213144],
    [0.0, 1.9380505347476888]
]

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'fixed_weight' MinimisationHandler class")

        for i, t_pdf_dict in enumerate(time_pdfs):

            llh_dict = {
                "llh_name": "standard",
                "llh_sig_time_pdf": t_pdf_dict,
                "llh_bkg_time_pdf": {
                    "time_pdf_name": "steady",
                },
                "llh_energy_pdf": {
                    "energy_pdf_name": "power_law"
                }
            }


            # Test three declinations

            unblind_dict = {
                "mh_name": "fixed_weights",
                "dataset": ps_v002_p01.get_seasons("IC79", "IC86_1"),
                "catalogue": catalogue,
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]
            self.assertEqual(list(res["x"]), true_parameters[i])

            logging.info("Best fit values", list(res["x"]))
            logging.info("Reference best fit", true_parameters)


if __name__ == '__main__':
    unittest.main()

"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack import analyse, ResultsHandler

logging.getLogger().setLevel("INFO")

catalogue = tde_catalogue_name("jetted")


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_full_chain(self):

        logging.info("Testing MinimisationHandler analysis chain")

        base_name = "tests/test_analysis_chain/"

        for j, gamma in enumerate([2.0, 2.5]):
            # Initialise Injectors/LLHs

            inj_dict = {
                "injection_sig_time_pdf": {
                    "time_pdf_name": "steady"
                },
                "injection_energy_pdf": {
                    "energy_pdf_name": "power_law",
                    "gamma": gamma
                }
            }

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

            # Test three declinations

            mh_dict = {
                "name": f"{base_name}/{gamma}/",
                "mh_name": "fixed_weights",
                "dataset": icecube_ps_3_year.get_seasons("IC86-2011"),
                "catalogue": catalogue,
                "inj_dict": inj_dict,
                "llh_dict": llh_dict,
                "n_steps": 5,
                "n_trials": 10,
                "scale": [3.,500.][j]
            }

            analyse(mh_dict, n_cpu=2, cluster=False)

            rh = ResultsHandler(mh_dict)

        ub_dict = dict(mh_dict)

        ub_dict["background_ts"] = base_name

        ub = create_unblinder(ub_dict, full_plots=True, scan_2d=True)

if __name__ == '__main__':
    unittest.main()

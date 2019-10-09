"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
import pickle
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from flarestack.data.icecube import ps_v002_p01
from flarestack.core.unblinding import create_unblinder
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.shared import energy_spline_dir

base_energy_pdf =     {
        "energy_pdf_name": "power_law",
        "gamma": 3.0
    }

g = EnergyPDF.create(base_energy_pdf)

n_steps = 1e3

e_range = np.logspace(0, 7, n_steps)

f = InterpolatedUnivariateSpline(e_range, np.log(g.f(e_range)))

spline_save_path = "{0}e_2_power_law_{1}.npy".format(energy_spline_dir, n_steps)

logging.info("Saving to {0}".format(spline_save_path))

with open(spline_save_path, "wb") as h:
    pickle.dump(f, h)

# Initialise Injectors/LLHs

energy_pdfs = [
    base_energy_pdf,
    {
        "energy_pdf_name": "spline",
        "spline_path": spline_save_path
    },
]

true_parameters = [
    [1.8538431668730444],
    [1.8033620932638081]
]

catalogue = ps_catalogue_name(0.4)


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        logging.info("Testing 'fixed_weight' MinimisationHandler class")

        for i, e_pdf_dict in enumerate(energy_pdfs):

            llh_dict = {
                "llh_name": "fixed_energy",
                "llh_sig_time_pdf": {
                    "time_pdf_name": "steady"
                },
                "llh_bkg_time_pdf": {
                    "time_pdf_name": "steady",
                },
                "llh_energy_pdf": e_pdf_dict
            }

            unblind_dict = {
                "mh_name": "fixed_weights",
                "dataset": ps_v002_p01.get_seasons("IC86_1"),
                "catalogue": ps_catalogue_name(0.6),
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]

            self.assertEqual(list(res["x"]), true_parameters[i])

            logging.info("Best fit values {0}".format(list(res["x"])))
            logging.info("Reference best fit {0}".format(true_parameters[i]))


if __name__ == '__main__':
    unittest.main()

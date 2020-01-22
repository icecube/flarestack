"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
import pickle
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.unblinding import create_unblinder
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.shared import energy_spline_dir

logger = logging.getLogger()
logger.setLevel("ERROR")

base_energy_pdf =     {
        "energy_pdf_name": "power_law",
        "gamma": 3.0
    }

g = EnergyPDF.create(base_energy_pdf)

n_steps = 1e3

e_range = np.logspace(0, 7, int(n_steps))

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
    # Outdated style, test for backwards-compatibility
    {
        "Name": "PowerLaw",
        "Gamma": 3.0
    },
    {
        "Name": "Spline",
        "Spline Path": spline_save_path
    }
]

true_parameters = [
    [1.35150508],
    [1.34119769],
    [1.35150508],
    [1.34119769]
]

catalogue = ps_catalogue_name(-0.5)


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
                "dataset": icecube_ps_3_year.get_seasons('IC79-2010', 'IC86-2011'),
                "catalogue": catalogue,
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]
            for j, x in enumerate(list(res["x"])):
                self.assertAlmostEqual(x, true_parameters[i][j], places=1)

            logging.info("Best fit values {0}".format(list(res["x"])))
            logging.info("Reference best fit {0}".format(true_parameters[i]))


if __name__ == '__main__':
    unittest.main()

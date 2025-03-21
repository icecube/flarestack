"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""

import logging
import unittest

import numpy as np
from astropy import units as u

from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.utils import calculate_astronomy, load_catalogue

true_res_astro = {
    "Energy Flux (GeV cm^{-2} s^{-1})": 1.151292546497023e-08,
}

catalogue = tde_catalogue_name("jetted")


class TestUtilAstro(unittest.TestCase):
    def setUp(self):
        pass

    def test_neutrino_astronomy(self):
        logging.info("Testing neutrino_astronomy util function.")

        injection_energy_pdf = {"energy_pdf_name": "power_law", "gamma": 2.0}

        cat = load_catalogue(catalogue)

        res_astro = calculate_astronomy(1.0e-9, injection_energy_pdf)

        logging.info("Calculated values {0}".format(res_astro))
        logging.info("Reference  values {0}".format(true_res_astro))

        for key in ["Energy Flux (GeV cm^{-2} s^{-1})"]:
            self.assertAlmostEqual(
                np.log(res_astro[key]), np.log(true_res_astro[key]), places=2
            )


if __name__ == "__main__":
    unittest.main()

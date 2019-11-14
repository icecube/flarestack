"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.utils import get_diffuse_flux_at_1GeV, get_diffuse_flux_at_100TeV, calculate_transient_cosmology
from flarestack.analyses.ccsn.stasik_2017.sn_cosmology import ccsn_clash_candels
from astropy import units as u

default_flux_100TeV = [
    (2.2333333333333334e-18/(u.cm ** 2 * u.GeV * u.s * u.sr), 2.5),
    (1.01e-18/(u.cm ** 2 * u.GeV * u.s * u.sr), 2.19),
    (1.44e-18/(u.cm ** 2 * u.GeV * u.s * u.sr), 2.29)
]

true_cosmology = 8.966193871269827e-07 / (u.cm ** 2 * u.GeV * u.s * u.sr)

class TestUtilCosmo(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_diffuse_flux(self):

        logging.info("Testing get_diffuse_flux util functions.")

        fits = ["joint_15", "northern_tracks_17", "northern_tracks_19"]

        for i, fit in enumerate(fits):
            logging.info("Testing fit: {0}".format(fit))

            res_100_tev = get_diffuse_flux_at_100TeV(fit)

            self.assertEqual(res_100_tev, default_flux_100TeV[i])

            expected_1_gev = res_100_tev[0] * (10.**5)**res_100_tev[1]

            res_1_gev = get_diffuse_flux_at_1GeV(fit)

            self.assertEqual(expected_1_gev, res_1_gev[0])

            self.assertEqual(res_100_tev[1], res_1_gev[1])

            logging.info("Calculated values {0}".format(res_100_tev))
            logging.info("Reference  values {0}".format(default_flux_100TeV))

    def test_neutrino_cosmology(self):

        fit = "joint_15"
        bestfit = get_diffuse_flux_at_100TeV(fit)

        e_pdf_dict = {
         "energy_pdf_name": "power_law",
         'source_energy_erg': 1.e48 * u.erg,
         "gamma": bestfit[1]
        }

        # Use Supernova rate

        res = calculate_transient_cosmology(
         e_pdf_dict, ccsn_clash_candels, "test_CCSN", zmax=8.0,
         diffuse_fit=fit
        )

        self.assertAlmostEqual(res.value, true_cosmology.value, delta=2)

if __name__ == '__main__':
    unittest.main()

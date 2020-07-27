import logging
import unittest
from astropy import units as u
from flarestack.cosmo import get_rate
from flarestack.cosmo.rates import source_maps
from flarestack.cosmo.rates.tde_rates import tde_evolutions, local_tde_rates
from flarestack.cosmo.rates.sfr_rates import sfr_evolutions, local_sfr_rates
from flarestack.cosmo.rates.ccsn_rates import kcc_rates, sn_types

class TestCosmoRates(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_rates(self):

        logging.info("Testing get_rates util functions.")

        for vals in source_maps.values():
            for val in vals:
                f = get_rate(val)
                f(1.0)

    def test_tde_rates(self):

        for evolution in tde_evolutions.keys():
            for rate in local_tde_rates.keys():
                get_rate("tde", evolution_name=evolution, rate_name=rate)

        f = get_rate("tde", evolution_name="biehl_18_jetted", m=-2)
        true = 2.e-07 / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0)/true, 1.0, delta=0.05)

    def test_sfr_rates(self):

        for evolution in sfr_evolutions.keys():
            for rate in local_sfr_rates.keys():
                get_rate("sfr", evolution_name=evolution, rate_name=rate)

        f = get_rate("sfr")
        true = 0.08687592762508031 * u.solMass / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0)/true, 1.0, delta=0.05)

    def test_ccsn_rates(self):

        for kcc_name in kcc_rates.keys():
            for sn_sublass in sn_types.keys():
                get_rate("ccsn", kcc_name=kcc_name, sn_sublass=sn_sublass)

        f = get_rate("ccsn", subclass="Ibc", fraction=0.5)
        true = 0.00029537815392527303 / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0)/true, 1.0, delta=0.05)

if __name__ == '__main__':
    unittest.main()
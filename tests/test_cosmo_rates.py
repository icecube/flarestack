import logging
import unittest

import numpy as np
from astropy import units as u

from flarestack.cosmo import get_rate
from flarestack.cosmo.rates import source_maps
from flarestack.cosmo.rates.ccsn_rates import kcc_rates, sn_subclass_rates
from flarestack.cosmo.rates.fbot_rates import local_fbot_rates
from flarestack.cosmo.rates.frb_rates import local_frb_rates
from flarestack.cosmo.rates.grb_rates import grb_evolutions, local_grb_rates
from flarestack.cosmo.rates.sfr_rates import local_sfr_rates, sfr_evolutions
from flarestack.cosmo.rates.tde_rates import local_tde_rates, tde_evolutions

zrange = np.linspace(0.0, 8.0, 5)


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
                f = get_rate("tde", evolution_name=evolution, rate_name=rate)
                f(zrange)

        f_mid, f_low, f_high = get_rate(
            "tde", evolution_name="biehl_18_jetted", m=-2, with_range=True
        )
        true = [
            2.0e-07 / (u.Mpc**3 * u.yr),
            1.0e-07 / (u.Mpc**3 * u.yr),
            3.0e-07 / (u.Mpc**3 * u.yr),
        ]

        for i, f in enumerate([f_mid, f_low, f_high]):
            self.assertAlmostEqual(f(1.0) / true[i], 1.0, delta=0.05)

    def test_sfr_rates(self):
        for evolution in sfr_evolutions.keys():
            for rate in local_sfr_rates.keys():
                f = get_rate("sfr", evolution_name=evolution, rate_name=rate)
                f(zrange)

        f = get_rate("sfr")
        true = 0.08687592762508031 * u.solMass / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0) / true, 1.0, delta=0.05)

    def test_ccsn_rates(self):
        for kcc_name in kcc_rates.keys():
            for subclass_fractions_name, (sn_type, _) in sn_subclass_rates.items():
                for sn_subclass in sn_type.keys():
                    f = get_rate(
                        "ccsn",
                        kcc_name=kcc_name,
                        sn_subclass=sn_subclass,
                        subclass_fractions_name=subclass_fractions_name,
                    )
                    f(zrange)

        f = get_rate("ccsn", sn_subclass="Ibc", fraction=0.5)

        true = 7.236764771169189e-05 / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0) / true, 1.0, delta=0.05)

    def test_grb_rates(self):
        for evolution in grb_evolutions.keys():
            for rate in local_grb_rates.keys():
                f = get_rate("grb", evolution_name=evolution, rate_name=rate)
                f(zrange)

        f_mid, f_low, f_high = get_rate(
            "grb", evolution_name="lien_14", with_range=True
        )
        true = [
            1.7635240284867526e-09 / (u.Mpc**3 * u.yr),
            1.6290956630551653e-09 / (u.Mpc**3 * u.yr),
            1.9705101110066845e-09 / (u.Mpc**3 * u.yr),
        ]

        for i, f in enumerate([f_mid, f_low, f_high]):
            self.assertAlmostEqual(f(1.0) / true[i], 1.0, delta=0.05)

    def test_fbot_rates(self):
        for evolution in sfr_evolutions.keys():
            for rate in local_fbot_rates.keys():
                f = get_rate("fbot", evolution_name=evolution, rate_name=rate)
                f(zrange)

        f = get_rate("fbot")
        true = 4.054209955837081e-06 / (u.Mpc**3 * u.yr)

        self.assertAlmostEqual(f(1.0) / true, 1.0, delta=0.05)

    def test_frb_rates(self):
        for evolution in sfr_evolutions.keys():
            for rate in local_frb_rates.keys():
                f = get_rate("frb", evolution_name=evolution, rate_name=rate)
                f(zrange)

        f_mid, f_low, f_high = get_rate("frb", with_range=True)
        true = [
            0.418741971152887 / (u.Mpc**3 * u.yr),
            0.0637090135917256 / (u.Mpc**3 * u.yr),
            0.9272557341850235 / (u.Mpc**3 * u.yr),
        ]

        for i, f in enumerate([f_mid, f_low, f_high]):
            self.assertAlmostEqual(f(1.0) / true[i], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()

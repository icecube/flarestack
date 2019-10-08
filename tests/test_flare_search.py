"""Test the flare search method using one fixed-seed background trial. The
process is deterministic, so the same flare should be found each time.
"""
import logging
from flarestack.data.icecube import ps_v002_p01
from flarestack.utils.prepare_catalogue import ps_catalogue_name
import unittest
from flarestack.core.unblinding import create_unblinder

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

llh_time = {
    "time_pdf_name": "custom_source_box"
}

unblind_llh = {
    "llh_name": "standard",
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
    "llh_energy_pdf": llh_energy
}


cat_path = ps_catalogue_name(-0.1)

unblind_dict = {
    "mh_name": "flare",
    "dataset": ps_v002_p01.get_seasons("IC79", "IC86_1"),
    "catalogue": cat_path,
    "llh_dict": unblind_llh
}

# Inspecting the neutrino lightcurve for this fixed-seed scramble confirms
# that the most significant flare is in a 14 day window. The best-fit
# parameters are shown below. As both the scrambling and fitting is
# deterministic, these values should be returned every time this test is run.

true_parameters = [
    4.097462956856254,
    2.533607210908644,
    55876.89316064464,
    55892.569503379375,
    14.084548753227864
]


class TestFlareSearch(unittest.TestCase):

    def setUp(self):
        pass

    def test_flare(self):
        logging.info("Testing 'flare' LLH class")
        ub = create_unblinder(unblind_dict)
        res = [x for x in ub.res_dict["Parameters"].values()]
        self.assertEqual(res, true_parameters)

        logging.info("Best fit values {0}".format(list(res)))
        logging.info("Reference best fit {0}".format(true_parameters))


if __name__ == '__main__':
    unittest.main()
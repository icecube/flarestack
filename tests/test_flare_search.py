"""Test the flare search method using one fixed-seed background trial. The
process is deterministic, so the same flare should be found each time.
"""
from __future__ import print_function
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
import unittest
from flarestack.core.unblinding import create_unblinder

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": 2.0,
}

llh_time = {
    "time_pdf_name": "FixedEndBox"
}

unblind_llh = {
    "name": "standard",
    "llh_time_pdf": llh_time,
    "llh_energy_pdf": llh_energy
}


cat_path = ps_catalogue_name(-0.1)

unblind_dict = {
    "mh_name": "flare",
    "datasets": ps_7year.get_seasons("IC79", "IC86_1"),
    "catalogue": cat_path,
    "llh_dict": unblind_llh
}

# Inspecting the neutrino lightcurve for this fixed-seed scramble confirms
# that the most significant flare is in a 14 day window. The best-fit
# parameters are shown below. As both the scrambling and fitting is
# deterministic, these values should be returned every time this test is run.

true_parameters = [
    4.363432792437096,
    2.648209203557114,
    55876.89316064464,
    55892.569503379375,
    14.084548753227864
]


class TestFlareSearch(unittest.TestCase):

    def setUp(self):
        pass

    def test_flare(self):
        print("\n")
        print("\n")
        print("Testing 'flare' LLH class")
        print("\n")
        print("\n")
        ub = create_unblinder(unblind_dict)
        res = [x for x in ub.res_dict["Parameters"].values()]
        self.assertEqual(res, true_parameters)

        print("Best fit values", list(res))
        print("Reference best fit", true_parameters)


if __name__ == '__main__':
    unittest.main()
"""Test the flare search method using one fixed-seed background trial. The
process is deterministic, so the same flare should be found each time. The
test is based on the flare search analysis of the TDE Swift J1644+57.
"""
import numpy as np
from flarestack.core.unblinding import Unblinder
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC79_dict
from flarestack.shared import catalogue_dir
import unittest

analyses = dict()

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

unblind_llh = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Negative n_s?": False,
    "Flare Search?": True
}

name = "analyses/tde/unblind_individual_tdes/SwiftJ1644+57"


cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
catalogue = np.load(cat_path)

unblind_dict = {
    "name": name,
    "datasets": [IC79_dict],
    "catalogue": cat_path,
    "llh kwargs": unblind_llh,
}

# Inspecting the neutrino lightcurve for this fixed-seed scramble confirms
# that the most significant flare is in a 2.5 day window. The best-fit
# parameters are shown below. As fitting is deterministic, these values
# should be returned every time this test is run.

true_parameters = [
    1.9803355734180155,
    2.218092168368481,
    55655.88411894165,
    55658.743028581,
    2.4960624171217205
]


class TestFlareSearch(unittest.TestCase):

    def setUp(self):
        pass

    def test_flare(self):
        ub = Unblinder(unblind_dict)
        res = ub.res_dict["Swift J1644+57"]["Parameters"]
        self.assertEqual(res, true_parameters)


if __name__ == '__main__':
    unittest.main()
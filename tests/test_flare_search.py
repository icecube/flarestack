"""Test the flare search method using one fixed-seed background trial. The
process is deterministic, so the same flare should be found each time.
"""
import numpy as np
from flarestack.core.unblinding import Unblinder
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.utils.prepare_catalogue import ps_catalogue_name
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

name = "tests/test_flare_search/"


cat_path = ps_catalogue_name(-0.1)
catalogue = np.load(cat_path)

unblind_dict = {
    "name": name,
    "datasets": [IC86_1_dict],
    "catalogue": cat_path,
    "llh kwargs": unblind_llh,
}

# Inspecting the neutrino lightcurve for this fixed-seed scramble confirms
# that the most significant flare is in a 14 day window. The best-fit
# parameters are shown below. As both the scrambling and fitting is
# deterministic, these values should be returned every time this test is run.

true_parameters = [
    4.470342243452785,
    2.6270936625866463,
    55876.89316064464,
    55892.569503379375,
    14.084548753227864
]


class TestFlareSearch(unittest.TestCase):

    def setUp(self):
        pass

    def test_flare(self):
        ub = Unblinder(unblind_dict)
        key = [x for x in ub.res_dict.iterkeys() if x != "TS"][0]
        res = ub.res_dict[key]["Parameters"]
        self.assertEqual(res, true_parameters)

        print "Best fit values", res
        print "Reference best fit", res


if __name__ == '__main__':
    unittest.main()
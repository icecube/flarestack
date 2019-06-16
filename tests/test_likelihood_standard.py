"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
from __future__ import print_function
import unittest
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.unblinding import create_unblinder

# Initialise Injectors/LLHs

llh_dict = {
    "name": "standard",
    "llh_time_pdf": {
        "time_pdf_name": "Steady"
    },
    "llh_energy_pdf": {
        "energy_pdf_name": "PowerLaw"
    }
}

# Loop over sin(dec) values

sindecs = np.linspace(0.5, -0.5, 3)


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [
    [2.852420453833985, 4.0],
    [0.0, 3.0128881793603277],
    [2.611395323800308, 2.797875532591461]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print("\n")
        print("\n")
        print("Testing 'standard' LLH class")
        print("\n")
        print("\n")

        # Test three declinations

        for j, sindec in enumerate(sindecs):

            unblind_dict = {
                "mh_name": "fixed_weights",
                "datasets": ps_7year.get_seasons("IC86_1"),
                "catalogue": ps_catalogue_name(sindec),
                "llh_dict": llh_dict,
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            res = ub.res_dict[key]
            self.assertEqual(list(res["x"]), true_parameters[j])

            print("Best fit values", list(res["x"]))
            print("Reference best fit", true_parameters[j])


if __name__ == '__main__':
    unittest.main()

"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
from __future__ import print_function
import unittest
import numpy as np
from flarestack.data.icecube import ps_v002_p01
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.asimov_estimator import AsimovEstimator

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
    [1.5659139129935208e-08],
    [8.596263924277218e-09],
    [1.1616079571847577e-07]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print("\n")
        print("\n")
        print("Testing AsimovEstimator")
        print("\n")
        print("\n")

        injection_energy = {
            "energy_pdf_name": "power_law",
            "gamma": 2.0,
        }

        injection_time = {
            "time_pdf_name": "Steady",
        }

        inj_dict = {
            "injection_energy_pdf": injection_energy,
            "injection_time_pdf": injection_time,
        }

        ae = AsimovEstimator(ps_v002_p01.get_seasons("IC86_1"), inj_dict)

        for i, sindec in enumerate(sindecs):
            cat_path = ps_catalogue_name(sindec)
            dp = ae.guess_discovery_potential(cat_path)
            self.assertEqual(dp, true_parameters[i])


if __name__ == '__main__':
    unittest.main()

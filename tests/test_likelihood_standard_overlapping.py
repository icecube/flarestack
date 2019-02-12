"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import unittest
import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.unblinding import create_unblinder

# Initialise Injectors/LLHs

llh_dict = {
    "name": "standard",
    "LLH Time PDF": {
        "Name": "Steady"
    },
    "LLH Energy PDF": {
        "Name": "Power Law"
    }
}

name = "tests/test_likelihood_spatial/"

# Loop over sin(dec) values

sindecs = np.linspace(0.5, -0.5, 3)


# These results arise from high-statistics sensitivity calculations,
# and can be considered the "true" answers. The results we obtain will be
# compared to these values.

true_parameters = [
    [2.7737611449101554, 4.0],
    [0.0, 2.9850865457146476],
    [2.5002777434622914, 2.7982700386928294]
]


class TestTimeIntegrated(unittest.TestCase):

    def setUp(self):
        pass

    def test_declination_sensitivity(self):

        print "\n"
        print "\n"
        print "Testing standard LLH class"
        print "\n"
        print "\n"

        # Test three declinations

        for i, sindec in enumerate(sindecs):
            subname = name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

            unblind_dict = {
                "name": subname,
                "mh_name": "fixed_weights",
                "datasets": [IC86_1_dict],
                "catalogue": ps_catalogue_name(sindec),
                "llh_dict": llh_dict,
                "llh kwargs": llh_dict
            }

            ub = create_unblinder(unblind_dict)
            key = [x for x in ub.res_dict.iterkeys() if x != "TS"][0]
            res = ub.res_dict[key]
            self.assertEqual(list(res["x"]), true_parameters[i])

            print "Best fit values", list(res["x"])
            print "Reference best fit", true_parameters[i]


if __name__ == '__main__':
    unittest.main()

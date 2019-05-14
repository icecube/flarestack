from __future__ import print_function
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.shared import flux_to_k
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.results import ResultsHandler
from flarestack.core.unblinding import create_unblinder
import unittest


llh_dict = {
    "name": "spatial",
    "llh_time_pdf": {
        "time_pdf_name": "Steady"
    }
}

name = "tests/test_likelihood_spatial/"

source = ps_catalogue_name(0.0)

unblind_dict = {
    "name": name,
    "mh_name": "fixed_weights",
    "datasets": [IC86_1_dict],
    "catalogue": ps_catalogue_name(0.5),
    "llh_dict": llh_dict
}

true_parameters = [2.065170712604553]


class TestSpatialLikelihood(unittest.TestCase):

    def setUp(self):
        pass

    def test_spatial(self):
        print("\n")
        print("\n")
        print("Testing spatial LLH class")
        print("\n")
        print("\n")

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters)

        print("Best fit values", list(res["x"]))
        print("Reference best fit", true_parameters)


if __name__ == '__main__':
    unittest.main()
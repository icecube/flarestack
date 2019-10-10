import logging
from flarestack.data.public import icecube_ps_3_year
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.core.unblinding import create_unblinder
import unittest

llh_dict = {
    "llh_name": "spatial",
    "llh_sig_time_pdf": {
        "time_pdf_name": "steady"
    },
    "llh_bkg_time_pdf": {
        "time_pdf_name": "steady"
    }
}

source = ps_catalogue_name(0.0)

unblind_dict = {
    "mh_name": "fixed_weights",
    "dataset": icecube_ps_3_year.get_seasons("IC86-2011"),
    "catalogue": ps_catalogue_name(0.5),
    "llh_dict": llh_dict
}

true_parameters = [1.9587621795637824]


class TestSpatialLikelihood(unittest.TestCase):

    def setUp(self):
        pass

    def test_spatial(self):
        logging.info("Testing 'spatial' LLH class")

        ub = create_unblinder(unblind_dict)
        key = [x for x in ub.res_dict.keys() if x != "TS"][0]
        res = ub.res_dict[key]
        self.assertEqual(list(res["x"]), true_parameters)

        logging.info("Best fit values {0}".format(list(res)))
        logging.info("Reference best fit {0}".format(true_parameters))


if __name__ == '__main__':
    unittest.main()
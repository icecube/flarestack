"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.utils import load_catalogue
from flarestack.data.public import icecube_ps_3_year
from flarestack.cosmo import simulate_transient_catalogue, get_rate
from astropy import units as u

default_cat = [
    (0.60035332, 0.01447306, 1., 1., 56327.97886837, 0., 0.,  5.74325505, b'src0'),
    (2.15873548, 0.61203329, 1., 1., 55571.83350285, 0., 0.,  8.16688538, b'src2'),
    (1.48028531, 0.5894872 , 1., 1., 56280.68081373, 0., 0., 12.91086896, b'src5'),
    (1.49408944, 0.57887806, 1., 1., 55794.73283546, 0., 0., 13.25596606, b'src6'),
    (4.6238958 , 1.01115431, 1., 1., 55470.64380879, 0., 0., 14.7840499 , b'src7'),
    (0.7948626 , 0.07284388, 1., 1., 55639.98487556, 0., 0., 15.42726226, b'src10'),
    (3.81169009, 0.74494309, 1., 1., 56189.61825254, 0., 0., 16.34293175, b'src11'),
    (1.48994508, 0.25250742, 1., 1., 56104.58603335, 0., 0., 16.85204117, b'src13'),
    (2.41173727, 0.1171684 , 1., 1., 56153.93605386, 0., 0., 17.63180755, b'src16'),
    (5.8434789 , 0.50270522, 1., 1., 55520.12711356, 0., 0., 19.56093283, b'src23')
]

inj_energy_dict = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0
}
inj_dict = {
    "injection_energy_pdf": inj_energy_dict
}

mh_dict = {
    "dataset": icecube_ps_3_year,
    "inj_dict": inj_dict
}

class TestSimulateCatalogue(unittest.TestCase):

    def setUp(self):
        pass

    def test_sim_catalogue(self):

        logging.info("Testing simulate_transient_catalogue util functions.")

        key = "strolger_15"

        ccsn_rate = get_rate("ccsn", evolution_name=key, rate_name=key)

        all_cat_names = simulate_transient_catalogue(mh_dict, ccsn_rate,
                                                     cat_name="test_sim_cat",
                                                     n_entries=3, seed=1111, resimulate=True
                                                     )

        cat = load_catalogue(all_cat_names["Northern"][0])

        print(cat)

        for i, x in enumerate(cat):

            for j, y in enumerate(list(tuple(x))):
                ref = list(default_cat[i])[j]
                if y == 0.0:
                    self.assertEqual(y, ref)
                elif isinstance(y, float):
                    self.assertAlmostEqual(y/ref, 1.0, delta=0.1)
                else:
                    self.assertEqual(y, ref)

        logging.info("Calculated values {0}".format(cat))
        logging.info("Reference  values {0}".format(default_cat))

if __name__ == '__main__':
    unittest.main()

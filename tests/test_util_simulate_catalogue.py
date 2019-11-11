"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import logging
import unittest
from flarestack.utils import simulate_transient_catalogue, load_catalogue
from flarestack.data.public import icecube_ps_3_year
from flarestack.analyses.ccsn.stasik_2017.sn_cosmology import ccsn_clash_candels
from astropy import units as u

default_cat = [
    (5.811969668623737, 0.5951779941241785, 1.0, 1.0, 55865.19521994358, 0.0, 0.0, 7.273894224497459, b'src1'),
    (2.158735484632992, 0.9257630103264483, 1.0, 1.0, 55596.00890708748, 0.0, 0.0, 12.910868955681929, b'src2'),
    (1.9507841593746549, 1.2728173161600764, 1.0, 1.0, 56240.71097641321, 0.0, 0.0, 14.803186034910702, b'src3'),
    (0.012628196837952205, 0.5024469314976999, 1.0, 1.0, 55368.17352301331, 0.0, 0.0, 16.129258470995595, b'src4'),
    (1.4940894422680904, 0.571487637553326, 1.0, 1.0, 55439.831085944126, 0.0, 0.0, 16.85204117243573, b'src6'),
    (4.62389580206626, 0.7687212846989784, 1.0, 1.0, 55631.71390012936, 0.0, 0.0, 16.914303579606642, b'src7'),
    (4.928689826216657, 0.21252021460922726, 1.0, 1.0, 55952.426964538616, 0.0, 0.0, 17.258380838262298, b'src9'),
    (0.7948625958469872, 0.848733248483472, 1.0, 1.0, 55594.026738938104, 0.0, 0.0, 17.631807549834043, b'src10'),
    (1.4899450798362572, 0.6534294282751264, 1.0, 1.0, 55512.10131154205, 0.0, 0.0, 18.55539016083994, b'src13'),
    (1.531033278273662, 0.14564702226791074, 1.0, 1.0, 56267.7738955578, 0.0, 0.0, 18.702089538347725, b'src15'),
    (2.4117372736588094, 0.05714316009441343, 1.0, 1.0, 56281.0560293483, 0.0, 0.0, 19.560932827381748, b'src16'),
    (4.004742499039263, 0.8754313371123236, 1.0, 1.0, 55880.50696163924, 0.0, 0.0, 22.49210507467782, b'src21'),
    (3.8382363895636513, 0.280975744358895, 1.0, 1.0, 55693.49508578242, 0.0, 0.0, 22.550623447724934, b'src22')
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

        all_cat_names = simulate_transient_catalogue(mh_dict, ccsn_clash_candels,
                                                     cat_name="test_sim_cat",
                                                     n_entries=3, seed=1111, resimulate=True
                                                     )

        cat = load_catalogue(all_cat_names["Northern"][0])

        for i, x in enumerate(cat):
            print(x.dtype.names)
            print(x)
            print(default_cat[i])
            for j, y in enumerate(tuple(x)):
                self.assertAlmostEqual(y, tuple(default_cat[i])[j])

        # self.assertEqual(res_100_tev[1], res_1_gev[1])

        logging.info("Calculated values {0}".format(cat))
        logging.info("Reference  values {0}".format(default_cat))

if __name__ == '__main__':
    unittest.main()

"""A standard time-integrated analysis is performed, using one year of
IceCube data (IC86_1).
"""
import unittest
import numpy as np
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack import create_unblinder

sindecs = np.linspace(0.9, 0.0, 4)

class TestSimcube(unittest.TestCase):

    def setUp(self):
        pass

    # def test_simcube(self):
    #
    #     print("\n")
    #     print("\n")
    #     print("Testing simulation of datasets")
    #     print("\n")
    #     print("\n")
    #
    #     from flarestack.data.simulate.simcube import simcube_dataset
    #
    #     bkg_e_pdf_dict = {
    #         "energy_pdf_name": "PowerLaw",
    #         "gamma": 3.7
    #     }
    #
    #     bkg_time_pdf_dict = {
    #         "time_pdf_name": "fixed_end_box",
    #         "start_time_mjd": 50000,
    #         "end_time_mjd": 50100
    #     }
    #
    #     simcube_dataset.set_sim_params(
    #         name="IC86-2012",
    #         bkg_time_pdf_dict=bkg_time_pdf_dict,
    #         bkg_flux_norm=1e8,
    #         bkg_e_pdf_dict=bkg_e_pdf_dict,
    #         resimulate=True
    #     )

    def test_dataset(self):

        print("\n")
        print("\n")
        print("Testing use of simulated datasets")
        print("\n")
        print("\n")

        # from flarestack.data.simulate.simcube import simcube_dataset
        #
        # # Initialise Injectors/LLHs
        #
        # llh_dict = {
        #     "name": "standard",
        #     "llh_time_pdf": {
        #         "time_pdf_name": "Steady"
        #     },
        #     "llh_energy_pdf": {
        #         "energy_pdf_name": "PowerLaw"
        #     }
        # }

        # Test three declinations

        # for j, sindec in enumerate(sindecs):

            # unblind_dict = {
            #     "mh_name": "fixed_weights",
            #     "datasets": simcube_dataset.get_seasons("IC86-2012"),
            #     "catalogue": ps_catalogue_name(sindec),
            #     "llh_dict": llh_dict,
            # }

            # ub = create_unblinder(unblind_dict)
            # key = [x for x in ub.res_dict.keys() if x != "TS"][0]
            # res = ub.res_dict[key]
            # # self.assertEqual(list(res["x"]), true_parameters[j])
            #
            # print("Best fit values", list(res["x"]))
            # print("Reference best fit", true_parameters[j])



if __name__ == '__main__':
    unittest.main()

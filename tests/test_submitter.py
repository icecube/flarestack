"""simle script to test the scale estimation implemented in the Submitter class"""

import unittest
import numpy as np
from flarestack.shared import flux_to_k
from flarestack.data.public import icecube_ps_3_year
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster.submitter import Submitter


injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

injection_time = {
    "time_pdf_name": "steady",
}

llh_time = {
    "time_pdf_name": "steady",
}

inj_kwargs = {
    "injection_energy_pdf": injection_energy,
    "injection_sig_time_pdf": injection_time,
}

llh_energy = injection_energy

llh_kwargs = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
}

base_name = "test/test_submitter/"

sindec = 0.
cat_path = ps_catalogue_name(sindec)
scale = flux_to_k(reference_sensitivity(sindec)) * 5
mh_dict = {
    "name": base_name,
    "mh_name": "fixed_weights",
    "dataset": icecube_ps_3_year,
    "catalogue": cat_path,
    "inj_dict": inj_kwargs,
    "llh_dict": llh_kwargs,
    "scale": scale
    # "n_trials": 50,
    # "n_steps": 10
}

public_sens_3yr = 4.533328532314386e-10
upper = 7.66510624e-12
lower = 7.93338706e-12


class TestSubmitter(unittest.TestCase):

    def setUp(self):
        pass

    def test_submitter(self):
        this_mh_dict = dict(mh_dict)
        this_mh_dict['name'] += 'test_submitter/'
        this_mh_dict['n_trials'] = 10
        this_mh_dict['n_steps'] = 3
        sb = Submitter.get_submitter(mh_dict, use_cluster=False)
        sb.analyse()

    def test_scale_estimation(self):
        this_mh_dict = dict(mh_dict)
        this_mh_dict['name'] += 'test_scale_estimation/'
        sb = Submitter.get_submitter(mh_dict, use_cluster=False, do_sensitivity_scale_estimation='quick_injections')
        sb.run_quick_injections_to_estimate_sensitivity_scale()
        estimated_sensitivity_scale = sb.mh_dict['scale']
        true_value = flux_to_k(public_sens_3yr / 0.5)
        self.assertAlmostEqual(estimated_sensitivity_scale, true_value, delta=0.1)


if __name__ == '__main__':
    unittest.main()
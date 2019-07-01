from builtins import str
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.core.minimisation import MinimisationHandler
from flarestack.shared import fs_scratch_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.tde.shared_TDE import tde_catalogue_name
from flarestack.core.results import ResultsHandler

# Initialise Injectors/LLHs

path = fs_scratch_dir + "tester_spline.npy"

injector_gamma = 2.0
sin_dec = 0.0

root_name = "analyses/angular_error_floor/check_bias/"

# for i, llh_name in enumerate(["spatial", "fixed_energy", "standard"]):
for i, llh_name in enumerate(["standard", "standard_overlapping"]):
    base_name = root_name + llh_name + "/"

    llh_dict = {
        "name": llh_name,
        "LLH Time PDF": {
            "Name": "Steady"
        }
    }

    if llh_name is not "spatial_overlapping":
        llh_dict["LLH Energy PDF"] = {
            "Name": "Power Law"
        }

    # for j, gamma in enumerate([1.5, 2.0, 3.0, 3.5]):
    for j, gamma in enumerate([2.]):
        name = base_name + str(gamma) + "/"

        if llh_name == "fixed_energy":
            llh_dict["LLH Energy PDF"]["Gamma"] = gamma

        inj_dict = {
            "Injection Time PDF": {
                "Name": "Steady"
            },
            "Injection Energy PDF": {
                "Name": "Power Law",
                "Gamma": gamma,
            },
            "fixed_n": 30
        }

        mh_dict = {
            "name": name,
            "mh_name": "fixed_weights",
            "datasets": [IC86_1_dict],
            # "catalogue": ps_catalogue_name(sin_dec),
            "catalogue": tde_catalogue_name("jetted"),
            "llh_dict": llh_dict,
            "inj kwargs": inj_dict
        }

        scale = flux_to_k(reference_sensitivity(sin_dec, gamma)) * 125 * (
            [4.0, 1.0, 0.3, 10.0][j]
        )

        mh = MinimisationHandler.create(mh_dict)
        mh.iterate_run(scale=scale, n_steps=2, n_trials=100)
        rh = ResultsHandler(mh_dict)
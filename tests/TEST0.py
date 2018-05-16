import numpy as np
import os
import cPickle as Pickle
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7_year import IC86_234_dict
from shared import plot_output_dir, flux_to_k, analysis_dir
from utils.prepare_catalogue import ps_catalogue_name
from utils.skylab_reference import skylab_7year_sensitivity
from scipy.interpolate import interp1d
from cluster import run_desy_cluster as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

max_window = 50.
flare_length = 10.

llh_time = {
    "Name": "FixedRefBox",
    "Fixed Ref Time (MJD)": 56100.,
    "Pre-Window": 0.,
    "Post-Window": max_window
}

llh_energy = injection_energy

no_flare = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Find Flare?": True
}

flare_no_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": False,
    "Flare Search?": True
}

flare_with_energy = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True
}

name = "tests/TEST0/"

sindec = 0.0

cat_path = ps_catalogue_name(sindec)

src_res = dict()

for i, llh_kwargs in enumerate([flare_no_energy]):

    label = ["Flare (fixed Gamma)", "Flare (fit Gamma)"][i]
    f_name = ["flare_fixed_gamma", "flare_fit_gamma"][i]

    flare_name = name + f_name + "/"

    res = dict()

    injection_time = dict(llh_time)
    injection_time["Post-Window"] = flare_length

    inj_kwargs = {
        "Injection Energy PDF": injection_energy,
        "Injection Time PDF": injection_time,
        "Poisson Smear?": True,
    }

    scale = flux_to_k(skylab_7year_sensitivity(sindec)) * (
            3500 / flare_length)

    mh_dict = {
        "name": flare_name,
        "datasets": [IC86_234_dict],
        "catalogue": cat_path,
        "inj kwargs": inj_kwargs,
        "llh kwargs": llh_kwargs,
        "scale": scale,
        "n_trials": 5,
        "n_steps": 5
    }

    analysis_path = analysis_dir + flare_name

    try:
        os.makedirs(analysis_path)
    except OSError:
        pass

    pkl_file = analysis_path + "dict.pkl"

    with open(pkl_file, "wb") as f:
        Pickle.dump(mh_dict, f)

    # rd.submit_to_cluster(pkl_file, n_jobs=2000)

    mh = MinimisationHandler(mh_dict)
    mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"], n_trials=50)

    res[flare_length] = mh_dict

    src_res[label] = res


for (f_type, res) in sorted(src_res.iteritems()):
    sens = []
    fracs = []

    for (length, rh_dict) in sorted(res.iteritems()):

        rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                            rh_dict["catalogue"], show_inj=True)
        sens.append(rh.sensitivity * float(length) * 60 * 60 * 24)
        fracs.append(float(length)/max_window)

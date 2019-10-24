"""Script to constrain TDE model from https://arxiv.org/abs/1904.07999,
with tabulated values kindly provided by Leonel Morejon.

These values are converted to a spline, and saved. This script is then run
to determine the sensitivity to these models.

These splines are injected, but a standard power law is fit.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.tde.shared_TDE import tde_catalogues, \
    tde_catalogue_name
from flarestack.cluster import run_desy_cluster as rd
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_seasons import custom_dataset
from flarestack.analyses.tde.tde_model_spline.convert_spline import tde_spline_output_path, min_nu_e_gev
from flarestack.core.minimisation import MinimisationHandler

analyses = dict()

# Initialise Injectors/LLHs


llh_energy = {
    "Name": "Power Law"
}

llh_time = {
    "Name": "FixedEndBox"
}

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Weights?": True
}

cat_path = tde_catalogue_name("jetted")
catalogue = np.load(cat_path)

name = "analyses/tde/test_model/"

injection_length = 100.

injection_time = llh_time = {
    "Name": "Box",
    "Pre-Window": 0.,
    "Post-Window": injection_length
}

# Inject a spline

injection_energy = {
    "Name": "Spline",
    "Spline Path": tde_spline_output_path,
    "E Min": 1. + min_nu_e_gev
}

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

scale = 1.5

mh_dict = {
    "name": name,
    "datasets": custom_dataset(txs_sample_v1, catalogue,
                               llh_kwargs["LLH Time PDF"]),
    "catalogue": cat_path,
    "inj kwargs": inj_kwargs,
    "llh kwargs": llh_kwargs,
    "scale": scale,
    "n_trials": 5,
    "n_steps": 15
}

pkl_file = make_analysis_pickle(mh_dict)

# rd.submit_to_cluster(pkl_file, n_jobs=10)

mh = MinimisationHandler(mh_dict)
mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"],
               n_trials=10)
mh.clear()

rh_dict = mh_dict

rh = ResultsHandler(rh_dict["name"],
                    rh_dict["llh kwargs"],
                    rh_dict["catalogue"],
                    show_inj=True
                    )

inj_time = injection_length * 60 * 60 * 24

astro_sens, astro_disc = rh.astro_values(
    rh_dict["inj kwargs"]["Injection Energy PDF"])

key = "Total Fluence (GeV cm^{-2} s^{-1})"

e_key = "Mean Luminosity (erg/s)"

sens_livetime = astro_sens[key] * inj_time
disc_livetime = astro_disc[key] * inj_time

sens_e = astro_sens[e_key] * inj_time
disc_e = astro_disc[e_key] * inj_time

print("Sens (int flux): {0}".format(rh.sensitivity * inj_time), rh.sensitivity)
print("SENS (livetime): {0}".format(sens_livetime))
print("Disc (livetime): {0}".format(disc_livetime))
print(sens_e, disc_e)


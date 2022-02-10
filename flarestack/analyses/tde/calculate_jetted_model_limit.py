"""Script to constrain TDE model from https://arxiv.org/abs/1904.07999,
(Figure 8) with tabulated values kindly provided by Leonel Morejon.

These values are converted to a spline, and saved. This script is then run
to determine the sensitivity to these models.

These splines are injected, but a standard power law is fit.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.core.energy_PDFs import EnergyPDF
from astropy.cosmology import WMAP9 as cosmo
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import (
    plot_output_dir,
    flux_to_k,
    make_analysis_pickle,
    k_to_flux,
)
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.tde.shared_TDE import (
    tde_catalogues,
    tde_catalogue_name,
    individual_tde_cat,
)
from flarestack.cluster import run_desy_cluster as rd
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_seasons import custom_dataset
from flarestack.analyses.tde.tde_model_spline.convert_spline import (
    tde_spline_output_path,
    min_nu_e_gev,
)
from flarestack.core.minimisation import MinimisationHandler
from astropy import units as u
from flarestack.analyses.tde.tde_cosmology import biehl_local_rate, measured_local_rate

analyses = dict()

# Initialise Injectors/LLHs


llh_energy = {"Name": "Power Law"}

llh_time = {"Name": "FixedEndBox"}

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Weights?": True,
}

cat_path = tde_catalogue_name("jetted")
# cat_path = individual_tde_cat("Swift J1644+57")
catalogue = np.load(cat_path)

name = "analyses/tde/test_model/"

injection_length = 100.0

injection_time = llh_time = {
    "Name": "Box",
    "Pre-Window": 0.0,
    "Post-Window": injection_length,
}

# Inject a spline

injection_energy = {
    "Name": "Spline",
    "Spline Path": tde_spline_output_path,
    "E Min": 1.0 + min_nu_e_gev,
}

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

scale = 1.5

mh_dict = {
    "name": name,
    "datasets": custom_dataset(txs_sample_v1, catalogue, llh_kwargs["LLH Time PDF"]),
    "catalogue": cat_path,
    "inj kwargs": inj_kwargs,
    "llh kwargs": llh_kwargs,
    "scale": scale,
    "n_trials": 5,
    "n_steps": 15,
}

pkl_file = make_analysis_pickle(mh_dict)

# rd.submit_to_cluster(pkl_file, n_jobs=10)
#
# mh = MinimisationHandler(mh_dict)
# mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"],
#                n_trials=100)
# mh.clear()
#
rh_dict = mh_dict

# Reference redshift from table 2

ref_redshift = 0.001
ref_distance = cosmo.luminosity_distance(ref_redshift)

print(
    "Reference distance from paper: z={0} ({1:.2f})".format(ref_redshift, ref_distance)
)

area = 4 * np.pi * (ref_distance.to("cm") ** 2)

print("Area at this distance is {0}".format(area))

energy_PDF = EnergyPDF.create(injection_energy)
e_integral = energy_PDF.fluence_integral() * u.GeV**2

print("Integral over energy is {0}".format(e_integral))

rh = ResultsHandler(
    rh_dict["name"], rh_dict["llh kwargs"], rh_dict["catalogue"], show_inj=True
)

inj_time = injection_length * 60 * 60 * 24 * u.s

astro_sens, astro_disc = rh.astro_values(rh_dict["inj kwargs"]["Injection Energy PDF"])

key = "Total Fluence (GeV cm^{-2} s^{-1})"

e_key = "Mean Luminosity (erg/s)"

sens_livetime = astro_sens[key] * inj_time * u.GeV / u.cm**2 / u.s
disc_livetime = astro_disc[key] * inj_time * u.GeV / u.cm**2 / u.s

sens_e = astro_sens[e_key] * inj_time * u.erg / u.s
disc_e = astro_disc[e_key] * inj_time * u.erg / u.s

flux_units = 1.0 / u.GeV / u.cm**2 / u.s

print("Sensitivity", rh.sensitivity)

print(
    "Sens (int flux): {0}".format(rh.sensitivity * inj_time * flux_units * e_integral),
    rh.sensitivity,
)
print("SENS (livetime): {0}".format(sens_livetime))
print("Disc (livetime): {0}".format(disc_livetime))
print(sens_e, disc_e)

print(sens_e)

# Constraint from https://arxiv.org/pdf/1711.03555.pdf
# Reference G best fit:
g = 540

# G is product of baryonic loading and rate / 0.1/gpc/year

baryonic_loading = (
    g * biehl_local_rate.to("Mpc-3 yr-1") / measured_local_rate.to("Mpc-3 yr-1")
)

baryonic_loading = 10.0

print("Baryonic Loading from Biehl best fit is: {0}".format(baryonic_loading))

# Upscaling factor is ratio of baryonic loading to assumed baryonic loading in https://arxiv.org/pdf/1711.03555.pdf
# (This is 10)

ratio = baryonic_loading * 0.1

print("This is {0} times larger than that assumed for the spline".format(ratio))

base_ratio = 0.1 / (e_integral / u.GeV**2)

conversion = baryonic_loading * base_ratio * area * e_integral / u.GeV / (u.cm**2)
print(conversion / area)
print(conversion.to("erg"))
print("Ratio is {0}".format(sens_e / conversion.to("erg")))

horizon = np.sqrt((conversion / area) / sens_livetime) * ref_distance
print("Horizon is {0:.1f}".format(horizon))

base_case = base_ratio * area * e_integral / u.GeV / (u.cm**2)

constraint = sens_e / base_case.to("erg")

print("Baryonic loading must be > {0}".format(constraint))

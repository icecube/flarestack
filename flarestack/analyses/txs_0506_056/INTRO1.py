from __future__ import print_function
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.core.minimisation import MinimisationHandler
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.analyses.txs_0506_056.make_txs_catalogue import \
    txs_catalogue, txs_cat_path

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

# Use a source that is constant in time

injection_time = {
    "Name": "Steady",
}

# Use a source with a spectral index of -2, with an energy range between
# 100 GeV and 10 Pev (10**7 GeV).

# injection_energy = {
#     "Name": "Spline",
#     "Spline Path": "/afs/ifh.de/user/s/steinrob/scratch/flarestack__data/tester_spline.npy"
# }

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Look for a source that is constant in time

llh_time = {
    "Name": "Steady",
}

# Try to fit a power law to the data

llh_energy = {
    "Name": "Power Law",
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
}

# Takes a guess at the correct flux scale, based on previous IceCube results

scale = flux_to_k(reference_sensitivity(np.sin(txs_catalogue["dec"]),
                  injection_energy["Gamma"])) * 70


# Assign a unique name for each different minimisation handler dictionary

name = "analyses/txs_0506_056/INTRO1/"

# Creates the Minimisation Handler dictionary, which contains all relevant
# information to run an analysis

mh_dict = {
    "name": name,
    "datasets": ps_7year[-2:-1],
    "catalogue": txs_cat_path,
    "inj kwargs": inj_kwargs,
    "llh kwargs": llh_kwargs,
    "scale": scale,
    "n_trials": 100,
    "n_steps": 10
}

# Creates a Minimisation Handler using the dictionary, and runs the trials

mh = MinimisationHandler(mh_dict)
mh.iterate_run(mh_dict["scale"], n_steps=mh_dict["n_steps"],
               n_trials=mh_dict["n_trials"])

# Creates a Results Handler to analyse the results, and calculate the
# sensitivity. This is the flux that needs to arrive at Earth, in order for
# IceCube to see an overfluctuation 90% of the time. Prints this information.

rh = ResultsHandler(mh_dict)
sens = rh.sensitivity

# Converts the flux at Earth to a required luminosity from the source,
# by scaling with distance and accounting for redshift

astro_sens, astro_disc = rh.astro_values(
    mh_dict["inj kwargs"]["Injection Energy PDF"])

# Print output

# Load the source
print("\n \n \n")
print("The source to be analysed is:")
print(txs_catalogue["Name"][0])

for field in txs_catalogue.dtype.names:
    print(field, ": \t", txs_catalogue[field][0])
print("\n")

print("FINAL RESULT", "\n")

print("Sensitivity is", sens, "GeV/cm^2")

print("This requires a neutrino luminosity of", end=' ')
print(astro_sens["Mean Luminosity (erg/s)"], "erg/s")

print("\n \n \n")

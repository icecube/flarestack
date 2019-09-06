from __future__ import print_function
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.data.icecube.ps_tracks.ps_v003_p02 import ps_v003_p02
from flarestack.shared import make_analysis_pickle
from flarestack.core.minimisation import MinimisationHandler
from flarestack.cluster import analyse, wait_cluster
from flarestack.analyses.txs_0506_056.make_txs_catalogue import \
    txs_catalogue, txs_cat_path

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

# Use a source that is constant in time

injection_time = {
    "time_pdf_name": "Steady",
}

# Use a source following a power law with a spectral index of -2, using the
# default energy range of 100 GeV to 10 Pev (10**7 GeV).

injection_energy = {
    "energy_pdf_name": "PowerLaw",
    "gamma": 2.0
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_kwargs = {
    "injection_energy_pdf": injection_energy,
    "injection_time_pdf": injection_time,
    "poisson_smear_bool": True,
}

# Set up the "likelihood" arguments, which determine how the fake data is
# analysed.

# Look for a source that is constant in time

llh_time = {
    "time_pdf_name": "Steady",
}

# Try to fit a power law to the data

llh_energy = {
    "energy_pdf_name": "PowerLaw",
}

# Set up a likelihood that fits the number of signal events (n_s), and also
# the spectral index (gamma) of the source

llh_kwargs = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_time_pdf": llh_time,
}


# Assign a unique name for each different minimisation handler dictionary

name = "analyses/txs_0506_056/INTRO1/"

# Creates the Minimisation Handler dictionary, which contains all relevant
# information to run an analysis

mh_dict = {
    "name": name,
    "mh_name": "fixed_weights",
    "datasets": ps_v002_p01.get_seasons("IC86_1"),
    "catalogue": txs_cat_path,
    "inj_dict": inj_kwargs,
    "llh_dict": llh_kwargs,
    "n_trials": 1,
    "n_steps": 10
}

mh = MinimisationHandler.create(mh_dict)
scale = mh.guess_scale()

mh_dict["scale"] = scale

# Creates a Minimisation Handler using the dictionary, and runs the trials

analyse(mh_dict,  n_cpu=2, n_jobs=1, cluster=False)
# wait_cluster()

# mh.iterate_run(scale, n_steps=mh_dict["n_steps"],
#                n_trials=mh_dict["n_trials"])

# Creates a Results Handler to analyse the results, and calculate the
# sensitivity. This is the flux that needs to arrive at Earth, in order for
# IceCube to see an overfluctuation 90% of the time. Prints this information.

rh = ResultsHandler(mh_dict)
sens = rh.sensitivity

# Converts the flux at Earth to a required luminosity from the source,
# by scaling with distance and accounting for redshift

astro_sens, astro_disc = rh.astro_values(
    mh_dict["inj_dict"]["injection_energy_pdf"])

# Print output

# Load the source
print("\n \n \n")
print("The source to be analysed is:")
print(txs_catalogue["source_name"][0])

for field in txs_catalogue.dtype.names:
    print(field, ": \t \t \t", txs_catalogue[field][0])
print("\n")

print("FINAL RESULT", "\n")

print("Sensitivity is", sens, "GeV/cm^2")
print("This requires a neutrino luminosity of", end=' ')
print(astro_sens["Mean Luminosity (erg/s)"], "erg/s")
print()
print("Discovery Potential is", rh.disc_potential, "GeV/cm^2")
print("(The discovery potential probably does not have good statistics)")
print()
print("REMINDER: our quick discovery potential estimate was:", mh.disc_guess,
      "GeV/cm^2")

print("\n \n \n")

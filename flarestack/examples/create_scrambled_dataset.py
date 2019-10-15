from flarestack.data.icecube import ps_v002_p01
from flarestack.utils import ps_catalogue_name, load_catalogue
import logging

logging.getLogger().setLevel("INFO")

# Initialise Injectors/LLHs

# Set up what is "injected" into the fake dataset. This is a simulated source

# Use a source that is emits as a box, beginning at MJD x, and lasting 0.01 days

injection_time = {
    "time_pdf_name": "fixed_ref_box",
    "fixed_ref_time_mjd": 56679.2044683, # Random time in ~2013
    "pre_window": 0., # Start 0 days before reference time
    "post_window": 0.01, # End 0.01 days after reference time
}

# Use a source following a power law with a spectral index of -2, using the
# default energy range of 100 GeV to 10 Pev (10**7 GeV).

injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
    "e_min_gev": 10**2,
    "e_max_gev": 10**7
}

# Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
# random variations in detected neutrino numbers

inj_dict = {
    "injection_energy_pdf": injection_energy,
    "injection_sig_time_pdf": injection_time,
    "poisson_smear_bool": True,
}

# Uses this season to create dataset

season = ps_v002_p01.get_single_season("IC86_234")

# Creates a dummy source at sin(dec)=0.1
# The source is saved in .npy format
# A catalogue of sources can be used instead, to simulate stacking

source_path = ps_catalogue_name(0.1)
sources = load_catalogue(source_path)

inj = season.make_injector(sources, **inj_dict)

# Flux scale is defined as 10^-9 Gev^-1 s^-1 sr^-1 cm^-2 at 1 GeV
# This flux is injected into the dataset, multiplied by the time pdf.

flux_scale = 1.

# Creates a dataset using scrambled background, with MC added on top

dataset = inj.create_dataset(flux_scale)

print(dataset)


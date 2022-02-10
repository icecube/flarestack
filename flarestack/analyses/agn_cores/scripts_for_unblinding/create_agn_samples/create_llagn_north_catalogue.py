"""
Script to create catalogue entries for LLAGN sample.

The catalogue is given by the positional cross-match between 2RXS and AllWISE,
and removing the 3LAC blazars. A Seyferntess PDF is assigned and only sources with
Seyfertness larger than 0.5 are selected in the final sample.
"""
from flarestack.analyses.agn_cores.shared_agncores import (
    raw_cat_dir,
    agn_catalogue_name,
    agn_cores_output_dir,
)
from shared_agncores import create_random_src, plot_catalogue
from flarestack.utils.prepare_catalogue import cat_dtype
import astropy.io.fits as pyfits
from astropy.table import Table
import numpy as np
import pandas as pd
import os


def select_nrandom_sources(cat, n_random=100):
    df = cat.to_pandas()
    df_random = df.sample(n=n_random)
    cat_new = Table.from_pandas(df_random)
    print(cat_new)
    return cat_new


def select_n_brightest_srcs(cat, nr_srcs):
    """
    Select the first nr_srcs brightest sources

    :param cat: original catalogue of sources
    :param nr_srcs: number of sources to select
    :return: catalogue after selection
    """
    print("Selecting", nr_srcs, "brightest sources.Length after cuts:", len(raw_cat))
    return cat[-nr_srcs:]


"""Open original (complete) catalogue"""
raw_cat = pyfits.open(
    raw_cat_dir + "LLAGN_2rxs2AllWiseSayfertness_no3LACbl_April2020_small.fits"
)
raw_cat = Table(raw_cat[1].data, masked=True)
print("Catalogue length:", len(raw_cat))

raw_cat = raw_cat[raw_cat["DEC_DEG"] > -5]  # Select Northen sky sources only
raw_cat = raw_cat.group_by("XRay_FLUX")  # order catalog by flux
print("Catalogue length after cut:", len(raw_cat))

new_cat = np.empty(len(raw_cat), dtype=cat_dtype)
new_cat["ra_rad"] = np.deg2rad(
    raw_cat["RA_DEG"]
)  # rosat RA in radians    #np.deg2rad(random_ra)
new_cat["dec_rad"] = np.deg2rad(
    raw_cat["DEC_DEG"]
)  # rosat DEC in radians  #np.deg2rad(random_dec)
new_cat["distance_mpc"] = np.ones(len(raw_cat))
new_cat["ref_time_mjd"] = np.ones(len(raw_cat))
new_cat["start_time_mjd"] = np.ones(len(raw_cat))
new_cat["end_time_mjd"] = np.ones(len(raw_cat))
new_cat["base_weight"] = raw_cat["XRay_FLUX"] * 1e13
new_cat["injection_weight_modifier"] = np.ones(len(raw_cat))

src_name = []
for src, vv10 in enumerate(raw_cat["2RXS_ID"]):
    # if (vv10!='N/A'):
    #     src_name.append(vv10)
    if raw_cat["2RXS_ID"][src] != "N/A":
        src_name.append(raw_cat["2RXS_ID"][src])
    elif raw_cat["XMMSL2_ID"][src] != "N/A":
        src_name.append(raw_cat["XMMSL2_ID"][src])
    else:
        print("No valid name found for source nr ", src)
        break
new_cat["source_name"] = src_name

save_path = agn_catalogue_name("lowluminosity", "irselected_north")

np.save(save_path, new_cat)

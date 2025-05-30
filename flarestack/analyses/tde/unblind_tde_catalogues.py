"""Script to unblind the TDE catalogues. Draws the background TS values
generated by compare_spectral_indices.py, in order to
quantify the significance of the result. Produces relevant post-unblinding
plots.
"""

import logging

import numpy as np

from flarestack.analyses.tde.shared_TDE import tde_catalogue_name, tde_catalogues
from flarestack.core.unblinding import create_unblinder
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.utils.custom_dataset import custom_dataset

logging.getLogger().setLevel("INFO")

analyses = dict()

# Initialise Injectors/LLHs

# Shared

llh_energy = {
    "energy_pdf_name": "power_law",
    "gamma": 2.0,
}

llh_time = {"time_pdf_name": "custom_source_box"}

llh_bkg_time = {"time_pdf_name": "steady"}

llh_dict = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": llh_bkg_time,
}

name_root = "analyses/tde/unblind_stacked_TDEs/"
bkg_ts_root = "analyses/tde/compare_spectral_indices/"

cat_res = dict()

res = []

for j, cat in enumerate(tde_catalogues):
    name = name_root + cat.replace(" ", "") + "/"

    logging.info(f"{name}")

    bkg_ts = bkg_ts_root + cat.replace(" ", "") + "/Fit Weights/"

    cat_path = tde_catalogue_name(cat)
    catalogue = load_catalogue(cat_path)

    unblind_dict = {
        "name": name,
        "mh_name": "fit_weights",
        "dataset": custom_dataset(
            txs_sample_v1, catalogue, llh_dict["llh_sig_time_pdf"]
        ),
        "catalogue": cat_path,
        "llh_dict": llh_dict,
        "background_ts": bkg_ts,
    }

    # ub = create_unblinder(unblind_dict, mock_unblind=False)
    ub = create_unblinder(unblind_dict, mock_unblind=False, disable_warning=True)

    r = ub.res_dict
    print(r)
    ns = np.zeros(len(catalogue))

    for x, val in r["Parameters"].items():
        if "n_s" in x:
            mask = np.array([str(y) in str(x) for y in catalogue["source_name"]])
            ns[mask] = float(val)
    print(f"nstot: {np.sum(ns)}")

    res.append((cat, ub.ts))

    print("Catalogue for latex export")

    print("Source & R.A & Dec & Distance & T$_{0}$ & T$_{1}$ & n$_{s}$ \\\\")
    print("& (deg.) & (deg.) & (Mpc) & (MJD) & (MJD) & \\\\")
    print("\hline")

    for i, x in enumerate(catalogue):
        print(
            f'{x["source_name"].decode()} & {np.degrees(x["ra_rad"]):.2f} & {np.degrees(x["dec_rad"]):.2f} & '
            f'{x["distance_mpc"]:.0f} & {x["start_time_mjd"]:.0f} & {x["end_time_mjd"]:.0f}  & {ns[i]:.2f} \\\\'
        )
    print("\hline")

for x in res:
    print(x)

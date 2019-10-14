from __future__ import print_function
import os
import numpy as np
import pickle as Pickle
from scipy.interpolate import interp1d
from flarestack.shared import limit_output_path
from astropy import units as u

ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_cat_dir = ccsn_cat_dir + "raw/"

# sn_cats = ["IIn", "IIp", "Ibc"]
sn_cats = ["IIn"]

sn_times = [100., 300., 1000.]
sn_times = [300.]


def sn_catalogue_name(sn_type, nearby=True):
    sn_name = sn_type + "_"

    if nearby:
        sn_name += "nearby.npy"
    else:
        sn_name += "distant.npy"

    return ccsn_cat_dir + sn_name


def sn_time_pdfs(sn_type):

    time_pdfs = []

    for i in sn_times:
        time_pdfs.append(
            {
                "time_pdf_name": "box",
                "pre_window": 20,
                "post_window": i
            }
        )

    if sn_type == "Ibc":
        time_pdfs.append(
            {
                "time_pdf_name": "box",
                "pre_window": 20,
                "post_window": 0
            }
        )

    return time_pdfs


def ccsn_limits(sn_type):

    base = "analyses/ccsn/stasik_2017/calculate_sensitivity/"
    path = base + sn_type + "/real_unblind/"

    savepath = limit_output_path(path)

    print("Loading limits from", savepath)
    with open(savepath, "r") as f:
        results = Pickle.load(f)
    return results


def ccsn_energy_limit(sn_type, gamma):
    results = ccsn_limits(sn_type)

    spline_y = np.exp(interp1d(results["x"], np.log(results["energy"]))(gamma))

    return spline_y * u.erg

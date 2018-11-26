import os
import cPickle as Pickle
from flarestack.shared import limit_output_path

ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_cat_dir = ccsn_cat_dir + "raw/"

sn_cats = ["IIn", "IIP", "Ibc"]


def sn_catalogue_name(sn_type, nearby=True):
    sn_name = sn_type + "_"

    if nearby:
        sn_name += "nearby.npy"
    else:
        sn_name += "distant.npy"

    return ccsn_cat_dir + sn_name


def sn_time_pdf(sn_type):

    if sn_type == "Ibc":
        time_pdf_dict = {
            "Name": "Box",
            "Pre-Window": 20,
            "Post-Window": 0
        }
    else:
        time_pdf_dict = {
            "Name": "Box",
            "Pre-Window": 20,
            "Post-Window": 300
        }

    return time_pdf_dict


def ccsn_limits(sn_type):

    base = "analyses/ccsn/calculate_sensitivity/"
    path = base + sn_type + "/real_unblind/"

    savepath = limit_output_path(path)

    print "Loading limits from", savepath
    with open(savepath, "r") as f:
        res_dict = Pickle.load(f)
    return res_dict

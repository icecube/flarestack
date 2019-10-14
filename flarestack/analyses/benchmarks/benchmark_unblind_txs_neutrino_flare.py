"""Script to replicate unblinding of the neutrino flare found for the source
TXS 0506+056, as described in https://arxiv.org/abs/1807.08794.
"""
from flarestack.core.unblinding import create_unblinder
from flarestack.data.icecube import txs_sample_v1
from flarestack.analyses.txs_0506_056.make_txs_catalogue import txs_cat_path,\
    txs_catalogue
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.analyses.txs_0506_056.load_gao_spectral_models import \
    spline_name

# Initialise Injectors/LLHs

# Shared

llh_time = {
    "Name": "FixedEndBox"
}

# llh_energy = {
#     "Name": "Power Law",
#     "Gamma": 2.18
# }

llh_energy = {
        "Name": "Spline",
        "Spline Path": spline_name(0)
}

unblind_llh = {
    "name": "fixed_energy",
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
}

name = "analyses/benchmarks/TXS_0506+056/"

unblind_dict = {
    "name": name,
    "mh_name": "fixed_weights",
    "datasets": custom_dataset(txs_sample_v1, txs_catalogue,
                               unblind_llh["LLH Time PDF"]),
    # "datasets": [IC86_234_dict],
    "catalogue": txs_cat_path,
    "llh_dict": unblind_llh,
}

ub = create_unblinder(unblind_dict, mock_unblind=False)

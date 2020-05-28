"""Script to replicate unblinding of the neutrino flare found for the source
TXS 0506+056, as described in https://arxiv.org/abs/1807.08794.
"""
from flarestack.core.unblinding import create_unblinder
from flarestack.data.icecube import txs_sample_v1, ps_v003_p02, nt_v002_p05
from flarestack.analyses.txs_0506_056.make_txs_catalogue import txs_cat_path,\
    txs_catalogue
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.analyses.txs_0506_056.load_gao_spectral_models import \
    spline_name

# Initialise Injectors/LLHs

# Shared

llh_time = {
    "time_pdf_name": "fixed_ref_box",
    "fixed_ref_time_mjd": 56937.81,
    "pre_window": 0.,
    "post_window": 57096.22 - 56937.81,
}
#
# llh_time = {
#     "time_pdf_name": "fixed_ref_box",
#     "fixed_ref_time_mjd": 56927.86,
#     "pre_window": 0.,
#     "post_window": 57116.76 - 56927.86,
# }

llh_energy = {
    "energy_pdf_name": "power_law",
}

unblind_llh = {
    "llh_name": "standard",
    "llh_energy_pdf": llh_energy,
    "llh_sig_time_pdf": llh_time,
}

name = "analyses/benchmarks/TXS_0506+056/"

unblind_dict = {
    "name": name,
    "mh_name": "fixed_weights",
    "dataset": custom_dataset(ps_v003_p02, txs_catalogue,
                               unblind_llh["llh_sig_time_pdf"]),
    # "dataset": txs_sample_v1.get_seasons(""),
    "catalogue": txs_cat_path,
    "llh_dict": unblind_llh,
}

ub = create_unblinder(unblind_dict, mock_unblind=False)

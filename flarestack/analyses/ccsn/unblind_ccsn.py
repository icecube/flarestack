import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.analyses.ccsn.shared_ccsn import sn_cats, sn_catalogue_name, \
    sn_time_pdfs
from flarestack.core.unblinding import create_unblinder
from flarestack.utils.custom_seasons import custom_dataset


name_root = "analyses/ccsn/unblind_ccsn/"
bkg_ts_root = "analyses/ccsn/calculate_sensitivity/"

llh_energy = {
    "Name": "Power Law",
}

res_dict = dict()

for cat in sn_cats:
    name = name_root + cat + "/"
    bkg_ts = bkg_ts_root + cat + "/"

    cat_path = sn_catalogue_name(cat)
    catalogue = np.load(cat_path)

    llh_times = sn_time_pdfs(cat)

    for llh_time in llh_times:
        unblind_llh = {
            "name": "standard",
            "LLH Energy PDF": llh_energy,
            "LLH Time PDF": llh_time,
        }

        unblind_dict = {
            "name": name,
            "mh_name": "fit_weights",
            "datasets": custom_dataset(ps_v002_p01, catalogue,
                                       llh_time),
            "catalogue": cat_path,
            "llh_dict": unblind_llh,
            "background TS": bkg_ts
        }

        ub = create_unblinder(unblind_dict, mock_unblind=False)

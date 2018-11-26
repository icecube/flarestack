import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.analyses.ccsn.shared_ccsn import sn_cats, sn_catalogue_name, \
    sn_time_pdf
from flarestack.core.unblinding import Unblinder
from flarestack.utils.custom_seasons import custom_dataset


name_root = "analyses/ccsn/unblind_ccsn/"
bkg_ts_root = "analyses/ccsn/calculate_sensitivity/"

llh_energy = {
    "Name": "Power Law",
}

for cat in sn_cats:

    name = name_root + cat + "/"
    bkg_ts = bkg_ts_root + cat + "/"

    cat_path = sn_catalogue_name(cat)
    catalogue = np.load(cat_path)

    llh_time = sn_time_pdf(cat)

    llh_kwargs = {
        "LLH Energy PDF": llh_energy,
        "LLH Time PDF": llh_time,
        "Fit Gamma?": True,
        "Fit Weights?": True
    }

    unblind_dict = {
        "name": name,
        "datasets": custom_dataset(ps_7year, catalogue,
                                   llh_time),
        "catalogue": cat_path,
        "llh kwargs": llh_kwargs,
        "background TS": bkg_ts
    }

    ub = Unblinder(unblind_dict, mock_unblind=False, full_plots=True)

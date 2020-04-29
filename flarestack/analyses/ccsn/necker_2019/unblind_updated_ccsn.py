import numpy as np
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import sn_cats, sn_times, updated_sn_catalogue_name, sn_time_pdfs
from flarestack.core.unblinding import create_unblinder
from flarestack.utils.custom_dataset import custom_dataset


name_root = "analyses/ccsn/necker_2019/unblind_ccsn"
bkg_ts_root = "analyses/ccsn/necker_2019/calculate_sensitivity/fit_weights"

llh_energy = {
    "energy_pdf_name": "power_law",
}

res_dict = dict()

for cat in sn_cats:

    cat_path = updated_sn_catalogue_name(cat)
    catalogue = np.load(cat_path)

    for pdf_type in sn_times[cat]:

        llh_times = sn_time_pdfs(cat, pdf_type=pdf_type)

        name = f'{name_root}/{pdf_type}/{cat}'
        bkg_ts = f'{bkg_ts_root}/{pdf_type}/{cat}'

        for llh_time in llh_times:

            time = llh_time['decay_time'] if 'decay' in pdf_type else \
                llh_time['pre_window'] + llh_time['post_window']

            unblind_llh = {
                "llh_name": "standard",
                "llh_energy_pdf": llh_energy,
                "llh_sig_time_pdf": llh_time,
            }

            unblind_dict = {
                "name": name,
                "ts_type": 'Fit Weights',
                "mh_name": "fit_weights",
                "dataset": custom_dataset(ps_v002_p01, catalogue, llh_time),
                "catalogue": cat_path,
                "llh_dict": unblind_llh,
                "background_ts": f'{bkg_ts}/{time}/'
            }

            ub = create_unblinder(unblind_dict, mock_unblind=True)

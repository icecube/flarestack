"""Script to calculate the sensitivity and discovery potential for CCSNe.
"""
import numpy as np
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.stasik_2017.shared_ccsn import sn_catalogue_name
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits, get_figure_limits
# from flarestack.analyses.ccsn.stasik_2017.calculate_sensitivity_reproduce_stasik import raw as original_raw
from flarestack.analyses.ccsn import get_sn_color
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
import math
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging
import time

# Set Logger Level

start = time.time()

logging.getLogger().setLevel("DEBUG")

logging.debug('logging level is DEBUG')

logging.getLogger('matplotlib').setLevel('INFO')

# LLH Energy PDF

llh_energy = {
    "energy_pdf_name": "power_law",
}

cluster = 100

# Spectral indices to loop over

# gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
# gammas = [1.8, 2.0, 2.5]
gammas = [1.8, 2, 2.5]
windows = [20, 15, 10, 5, 1]

# Base name

mh_name = 'fit_weights'

raw = f"analyses/ccsn/stasik_2017/estimate_improvement/{mh_name}/"

job_ids = []
# Loop over SN catalogues

plot_results = ''
cat = 'Ibc'
cat_res = dict()

name = raw + cat + "/"

logging.debug('name is ' + name)

cat_path = sn_catalogue_name(cat)
logging.debug('catalogue path: ' + str(cat_path))
catalogue = np.load(cat_path)

logging.debug('catalogue dtype: ' + str(catalogue.dtype))

closest_src = np.sort(catalogue, order="distance_mpc")[0]

# Loop over time PDFs

for window in windows:

    llh_time = {
        "time_pdf_name": "box",
        "pre_window": window,
        "post_window": 0
    }

    logging.debug(f'time_pdf is {llh_time}')

    time_key = str(window)

    time_res = dict()

    llh_dict = {
        "llh_name": "standard",
        "llh_energy_pdf": llh_energy,
        "llh_sig_time_pdf": llh_time,
        "llh_bkg_time_pdf": {
            "time_pdf_name": "steady"
        }
    }

    injection_time = llh_time

    # Loop over spectral indices

    for gamma in gammas:

        full_name = name + str(gamma) + "/" + time_key + '/'

        length = float(time_key)

        scale = (flux_to_k(reference_sensitivity(
            np.sin(closest_src["dec_rad"]), gamma=gamma
        ) * 40 * math.sqrt(float(len(catalogue)))) * 200.) / length / 4

        injection_energy = dict(llh_energy)
        injection_energy["gamma"] = gamma

        inj_dict = {
            "injection_energy_pdf": injection_energy,
            "injection_sig_time_pdf": injection_time,
            "poisson_smear_bool": True,
        }

        mh_dict = {
            "name": full_name,
            "mh_name": mh_name,
            "dataset": custom_dataset(ps_v002_p01, catalogue,
                                      llh_dict["llh_sig_time_pdf"]),
            "catalogue": cat_path,
            "inj_dict": inj_dict,
            "llh_dict": llh_dict,
            "scale": scale,
            "n_trials": 2000/cluster if cluster else 1000,
            "n_steps": 10
        }
        # !!! number of total trial is ntrials * n_jobs !!!

        job_id = None
        # job_id = analyse(mh_dict, cluster=True if cluster else False, n_cpu=1 if cluster else 32, n_jobs=cluster)
        job_ids.append(job_id)

        time_res[gamma] = mh_dict

    cat_res[time_key] = time_res


# Wait for cluster. If there are no cluster jobs, this just runs
if cluster and np.any(job_ids):
    logging.info(f'waiting for jobs {job_ids}')
    wait_for_cluster(job_ids)

# plot results

for gamma in gammas:

    logging.info('getting results for gamma = {:.2f}'.format(gamma))
    this_sens = list()
    this_sens_time = list()

    for time_key, time_res in cat_res.items():

        logging.info('getting results for t = {:.0f}'.format(float(time_key)))

        rh_dict = cat_res[time_key][gamma]
        this_rh = ResultsHandler(rh_dict)

        inj = rh_dict["inj_dict"]["injection_sig_time_pdf"]
        injection_length = inj["pre_window"] + inj["post_window"]
        inj_time = injection_length * 60 * 60 * 24

        astro_sens, astro_disc = this_rh.astro_values(
            rh_dict["inj_dict"]["injection_energy_pdf"])

        key = "Energy Flux (GeV cm^{-2} s^{-1})"

        this_sens.append((float(time_key), this_rh.sensitivity, this_rh.sensitivity_err))
        this_sens_time.append((float(time_key), astro_sens[key] * inj_time))

    this_sens = np.array(this_sens)
    this_sens_time = np.array(this_sens_time)

    logging.debug('plotting combined results')

    fig, ax = plt.subplots()

    errs = [[t[0] for t in this_sens[:,2]],[t[1] for t in this_sens[:,2]]]
    ax.errorbar(this_sens[:,0], this_sens[:,1], yerr=errs,
                ls='', marker='o', markersize=2, label='improved explosion time', color='orange', capsize=4)

    ax.invert_xaxis()
    ax.set_xlabel(r'$\Delta T$')
    ax.set_ylabel(r'Sensitivity Flux [$GeV^{-1} \, s^{-1} \, cm^{-2}$]')
    ax.set_title(r'$\gamma = {:.1f}$'.format(gamma))

    ax.legend()

    plot_dir = plot_output_dir(name + str(gamma) + '/')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    fn = plot_dir + 'estimated_improvement.pdf'
    logging.debug('saving plot to ' + fn)
    fig.savefig(fn)
    plt.close()

    # making fluence plot
    logging.debug('plotting fluence')
    fig, ax = plt.subplots()
    ax.plot(this_sens_time[:,0], this_sens_time[:,1],
            color='orange', ls='--', marker='o')
    ax.set_xlabel(r'$\Delta T$')
    ax.set_ylabel(r'Total Fluence [GeV \, cm$^{-2}$]')
    ax.set_title(r'$\gamma = {:.1f}$'.format(gamma))
    ax.invert_xaxis()
    fn = plot_dir + 'estimated_improvement_fluence.pdf'
    logging.debug('saving plot to ' + fn)
    fig.savefig(fn)
    plt.close()

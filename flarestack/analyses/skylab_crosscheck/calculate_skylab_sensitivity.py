"""Script to calculate the sensitivity and discovery potential for some sources to check consistency with skylab.
"""
import numpy as np
from flarestack.analyses.skylab_crosscheck.make_sources import fs_sources, nsources, same_sindecs
from flarestack.analyses.skylab_crosscheck.skylab_results import sl_data_dir
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import analyse, wait_for_cluster
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging

logging.getLogger().setLevel("DEBUG")
logging.debug('logging level is DEBUG')
logging.getLogger('matplotlib').setLevel('INFO')

full_res = dict()
analyses = dict()

mh_name = 'large_catalogue'
raw = "analyses/skylab_crosscheck/" + mh_name + '/'
data_dir = os.environ['HOME'] + '/flarestack_cc/'

same_sindecs = [0.25, 0.5, 0.75]

cluster = 100

# Initialise Injectors/LLHs

gamma = 2.0

injection_energy = {
    "energy_pdf_name": "power_law",
    "gamma": gamma,
}

llh_time = {
    "time_pdf_name": "steady",
}

injection_time = {
    "time_pdf_name": "steady",
}

llh_dict = {
    "llh_name": "standard_matrix",
    "llh_energy_pdf": injection_energy,
    "llh_sig_time_pdf": llh_time,
    "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
}

inj_dict = {
    "injection_energy_pdf": injection_energy,
    "injection_sig_time_pdf": injection_time,
}

length = 0
for season_name, season in ps_v002_p01.seasons.items():
    length += season.get_time_pdf().get_livetime()

logging.info('injection length in livetime is {:.2f}'.format(length))

sin_res = dict()

job_ids = list()

for sindec in same_sindecs:

    full_res = dict()

    for i, n in enumerate(nsources):

        logging.info(f'stacking {n} sources')
        logging.info(f'cat path is {fs_sources(i, sindec)}')

        name = raw + '{:.4f}/'.format(sindec) + str(n) + 'sources' if sindec is not None \
            else raw + 'None/' + str(n) + 'sources'
        catalogue = np.load(fs_sources(i, sindec))
        closest_src = np.sort(catalogue, order="distance_mpc")[0]

        scale = (flux_to_k(reference_sensitivity(
            np.sin(closest_src["dec_rad"]), gamma=gamma
        ) * 40 * (math.log(float(len(catalogue)), 4) + 1)) * 200.) / length

        mh_dict = {
            "name": name,
            "mh_name": mh_name,
            "dataset": custom_dataset(ps_v002_p01, catalogue,
                                      llh_dict["llh_sig_time_pdf"]),
            "catalogue": fs_sources(i, sindec),
            "inj_dict": inj_dict,
            "llh_dict": llh_dict,
            "scale": scale,
            "n_trials": 5 if n < 200 else 1,
            "n_steps": 10
        }

        job_id = None
        # job_id = analyse(mh_dict,
        #                  cluster=cluster,
        #                  n_cpu=1 if cluster and n < 200 else 4 if cluster and n > 200 else 32,
        #                  n_jobs=100 if n < 200 else 500)
        job_ids.append(job_id)

        full_res[str(n)] = mh_dict

    sin_res[str(sindec)] = full_res

if cluster and np.any(job_ids):
    logging.info(f'waiting for jobs {job_ids}')
    wait_for_cluster(job_ids)

for sindec in same_sindecs:

    full_res = sin_res[str(sindec)]

    sens = [[], []]

    for i, n in enumerate(nsources):

        rh_dict = full_res[str(n)]
        rh = ResultsHandler(rh_dict)

        sens[0].append(n)
        sens[1].append(rh.sensitivity)

    # load results from skylab if they exist
    skylab_result_path = sl_data_dir(sindec) + '/nsources_gamma{:.1f}.npy'.format(gamma)
    skylab_results = np.load(skylab_result_path) if os.path.isfile(skylab_result_path) else None
    skylab_results = skylab_results[[n in nsources for n in skylab_results['nsources']]]

    # normalize all points to
    norm_to = reference_sensitivity(sindec, gamma=gamma)

    fig, ax = plt.subplots()

    Nsqrt = np.sqrt(sens[0])
    Nflat = [1.] * len(sens[0])

    ax.plot(sens[0], Nsqrt, 'k--', label=r'$F \sim \sqrt{N}$')
    ax.plot(sens[0], Nflat, 'k-.', label=r'$F = const$')
    ax.plot(sens[0], np.array(sens[1])/sens[1][0], 'o-', label='flarestack')

    if skylab_results is not None:
        logging.info('drawing skylab results')
        ax.plot(skylab_results['nsources'],
                skylab_results['sensitivity']/skylab_results['sensitivity'][0],
                'o-', label='skylab')
    else:
        logging.info('no skylab results')

    ax.set_xlabel('$N$')
    ax.set_ylabel(r"$F/F_{single \, source}$")
    ax.set_xscale('log')
    ax.set_title('stacked sensitivity \n' + r'$\sin(\delta)=${:.2f}'.format(sindec))
    ax.legend()

    plt.tight_layout()

    fig.savefig(plot_output_dir(raw + '/{:.4f}/'.format(sindec)) + f'sens_nsources_gamma={gamma}.pdf')
    plt.close()

    fig2, ax2 = plt.subplots()

    Nsqrt = 1/np.sqrt(sens[0])
    Nrez = 1/np.array(sens[0])

    ax2.plot(sens[0], Nsqrt, 'k--', label=r'$F \sim \sqrt{N}$')
    ax2.plot(sens[0], Nrez, 'k-.', label=r'$F = const$')
    ax2.plot(sens[0], np.array(sens[1]) / sens[1][0] / np.array(sens[0]), 'o-', label='flarestack')

    if skylab_results is not None:
        logging.info('drawing skylab results')
        ax2.plot(skylab_results['nsources'], skylab_results['sensitivity'] / skylab_results['sensitivity'][0]\
                 / np.array(sens[0])
                , 'o-', label='skylab')
    else:
        logging.info('no skylab results')

    ax2.set_xlabel('$N$')
    ax2.set_ylabel(r"$F/F_{single \, source}/N$")
    ax2.set_xscale('log')
    ax2.set_title('stacked sensitivity per source\n' + r'$\sin(\delta)=${:.2f}'.format(sindec))
    ax2.legend()

    plt.tight_layout()

    fig2.savefig(plot_output_dir(raw + '/{:.4f}/'.format(sindec)) + f'sens_nsources_gamma={gamma}_persource.pdf')
    plt.close()

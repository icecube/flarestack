"""Script to calculate the sensitivity and discovery potential for some sources to check consistency with skylab.
"""
import numpy as np
from flarestack.analyses.skylab_crosscheck.make_sources import fs_sources, nsources
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

weights = 'fixed'
raw = "analyses/skylab_crosscheck/calculate_sensitivity/" + weights + '_weights/'
data_dir = os.environ['HOME'] + '/flarestack_cc/'

cluster = False


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
    "llh_name": "standard",
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

full_res = dict()

job_ids = list()
for i, n in enumerate(nsources):

    logging.info(f'stacking {n} sources')
    logging.info(f'cat path is {fs_sources(i)}')

    name = raw + str(n) + 'sources'
    catalogue = np.load(fs_sources(i))
    closest_src = np.sort(catalogue, order="distance_mpc")[0]

    scale = (flux_to_k(reference_sensitivity(
        np.sin(closest_src["dec_rad"]), gamma=gamma
    ) * 40 * math.sqrt(float(len(catalogue)))) * 200.) / length

    mh_dict = {
        "name": name,
        "mh_name": f"{weights}_weights",
        "dataset": custom_dataset(ps_v002_p01, catalogue,
                                  llh_dict["llh_sig_time_pdf"]),
        "catalogue": fs_sources(i),
        "inj_dict": inj_dict,
        "llh_dict": llh_dict,
        "scale": scale,
        "n_trials": 50,
        "n_steps": 10
    }

    job_id = None
    # job_id = analyse(mh_dict, cluster=cluster, n_cpu=1 if cluster else 32, n_jobs=100)
    job_ids.append(job_id)

    full_res[str(n)] = mh_dict

if cluster:
    logging.info(f'waiting for jobs {job_ids}')
    wait_for_cluster(job_ids)

# sens_livetime = [[], []]
sens = [[], []]

for i, n in enumerate(nsources):

    rh_dict = full_res[str(n)]
    rh = ResultsHandler(rh_dict)

    sens[0].append(n)
    sens[1].append(rh.sensitivity)

    # astro_sens, astro_disc = rh.astro_values(
    #     rh_dict["inj_dict"]["injection_energy_pdf"])
    # logging.debug(f'astro_sens.keys(): {astro_sens.keys()}')
    #
    # key = "Energy Flux (GeV cm^{-2} s^{-1})"
    # sens_livetime[1].append(astro_sens[key] * length * 60 * 60 * 24)
    # sens_livetime[0].append(n)

# load results from skylab if they exist
skylab_result_path = os.environ['HOME'] + \
                 '/scratch/skylab_scratch/skylab_output/data/nsources_gamma{:.1f}.npy'.format(gamma)
skylab_results = np.load(skylab_result_path) if os.path.isfile(skylab_result_path) else None

fig, ax = plt.subplots()

ax.plot(sens[0], sens[1], 'o-', label='flarestack')

if skylab_results is not None:
    logging.info('drawing skylab results')
    ax.plot(skylab_results['nsources'], skylab_results['sensitivity'], 'o-', label='skylab')

ax.set_xlabel('# of sources')
ax.set_ylabel(r"Energy Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
ax.set_xscale('log')
ax.set_title('stacked sensitivity')
ax.legend()

plt.tight_layout()

fig.savefig(plot_output_dir(raw) + f'sens_nsources_gamma={gamma}.pdf')
plt.close()

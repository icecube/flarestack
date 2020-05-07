import numpy as np
from astropy.table import Table
import argparse
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube import ps_v002_p01
from flarestack.shared import plot_output_dir, flux_to_k
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits, get_figure_limits, p_vals
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import sn_cats, updated_sn_catalogue_name, \
    sn_time_pdfs, raw_output_dir, pdf_names, limit_sens
from flarestack.analyses.ccsn.necker_2019.skylab_crosscheck_data import skylab_data
from flarestack.analyses.ccsn import get_sn_color
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
import math
import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flarestack.utils.custom_dataset import custom_dataset
import os
import logging
import time


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--analyse', type=bool, default=False)
args = parser.parse_args()


logging.getLogger().setLevel("DEBUG")
logging.debug('logging level is DEBUG')
logging.getLogger('matplotlib').setLevel('INFO')


injection_energy = {
    "energy_pdf_name": "power_law"
}

injection_time = {
    'time_pdf_name': 'steady'
}

llh_energy = {
    "energy_pdf_name": "power_law"
}

llh_time = {
    'time_pdf_name': 'steady'
}

cluster = 100

# Spectral indices to loop over
gammas = [2.]

# set common ref time mjd
ref_time_mjd = np.nan

# Base name
mh_name = 'fixed_weights'
northern = True
weighted = False
ps = False
raw = raw_output_dir + f"/skylab_crosscheck"

if northern:
    raw += '_northern'

if weighted:
    raw += '_weighted'

if ps:
    raw += '_ps'

if not os.path.exists(plot_output_dir(raw)):
    os.mkdir(plot_output_dir(raw))

full_res = dict()

job_ids = []

# Loop over SN catalogues
if __name__ == '__main__':

    sindec = dict()

    for cat in ['IIn']:#sn_cats:

        name = f'{raw}/{cat}'
        cat_path = updated_sn_catalogue_name(cat)
        catalogue = np.load(cat_path)
        catalogue['ref_time_mjd'] = [ref_time_mjd] * len(catalogue)
        logging.debug(f'{Table(catalogue)}')

        if northern:
            catalogue = catalogue[catalogue['dec_rad'] > 0]
            logging.debug(f'{Table(catalogue)}')

        if ps:
            # uncomment to NOT use only the closest source!!!
            i = np.argmin(catalogue['distance_mpc'])
            catalogue = catalogue[i:i+1]
            sindec[cat] = np.sin(catalogue['dec_rad'])
            logging.debug(f'{Table(catalogue)}')

        if not weighted:
            catalogue['distance_mpc'] = np.array([1] * len(catalogue))
            w = [1] * len(catalogue)
        else:
            w = 1/catalogue['distance_mpc']**2
            w = w/sum(w)

        logging.debug(f'{Table(catalogue)}')

        this_cat_path = f'{cat_path.split(".npy")[0]}_used.npy'
        np.save(this_cat_path, catalogue)

        logging.info(f'max distance = {max(catalogue["distance_mpc"])}')

        fig, ax = plt.subplots()
        for ra, dec, w in zip(catalogue['ra_rad'], catalogue['dec_rad'], w):
            ax.scatter(ra, dec, alpha=w, color=get_sn_color(cat))
        ax.scatter(catalogue['ra_rad'], catalogue['dec_rad'], edgecolors=get_sn_color(cat), facecolors='none')
        ax.set_xlabel('RA in rad')
        ax.set_ylabel('DEC in rad')
        ax.set_title('SNe ' + cat)
        fn = plot_output_dir(f'{raw}/{cat}_skymap.pdf')
        logging.debug("saving figure under {0}".format(fn))
        fig.savefig(fn)

        closest_src = np.sort(catalogue, order="distance_mpc")[0]
        cat_res = dict()

        for gamma in gammas:

            injection_energy['gamma'] = gamma

            inj_dict = {
                'injection_energy_pdf': injection_energy,
                'injection_sig_time_pdf': injection_time
            }

            llh_dict = {
                "llh_name": "standard",
                "llh_energy_pdf": llh_energy,
                "llh_sig_time_pdf": llh_time,
                "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
            }

            full_name = f'{name}/{gamma:.2f}'

            length = 365 * 7

            scale = 0.5 * (flux_to_k(reference_sensitivity(
                np.sin(closest_src["dec_rad"]), gamma=gamma
            ) * 40 * math.sqrt(float(len(catalogue)))) * 200.) / length

            if ps:
                scale *= 2
                if cat == 'IIn':
                    scale *= 1.5

            if cat == 'IIn':
                scale *= 0.3
            if cat == 'IIP':
                scale *= 1

            if not northern and weighted:
                if cat == 'IIn':
                    scale *= 1.5
                if cat == 'IIP':
                    scale *= 0.4

            mh_dict = {
                "name": full_name,
                "mh_name": mh_name,
                "dataset": custom_dataset(ps_v002_p01, catalogue, llh_dict["llh_sig_time_pdf"]),
                "catalogue": this_cat_path,
                "inj_dict": inj_dict,
                "llh_dict": llh_dict,
                "scale": scale,
                "n_trials": 100 / cluster if cluster else 500,
                "n_steps": 12
            }

            job_id = None
            if args.analyse:
                job_id = analyse(mh_dict,
                                 cluster=True if cluster else False,
                                 n_cpu=1 if cluster else 32,
                                 n_jobs=cluster,
                                 h_cpu='00:59:59')
            job_ids.append(job_id)

            cat_res[gamma] = mh_dict

        full_res[cat] = cat_res

    if cluster and np.any(job_ids):
        logging.info(f'waiting for jobs {job_ids}')
        wait_for_cluster(job_ids)

    stacked_sens_flux = {}

    # calculating sensitivities

    for cat_name, cat_res in full_res.items():

        stacked_sens_flux[cat_name] = dict()

        for gamma, rh_dict in cat_res.items():

            rh = ResultsHandler(rh_dict)

            stacked_sens_flux[cat_name][gamma] = {
                'sens': rh.sensitivity,
                'sens_e': rh.sensitivity_err
            }

    # making plots

    figaxs = {}

    for cat_name, cat_dict in stacked_sens_flux.items():

        for gamma, sens_dict in cat_dict.items():

            fs_label = ''
            sl_label = ''
            ref_label = ''
            if gamma not in figaxs:
                figaxs[gamma] = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                fs_label = 'Flarestack'
                sl_label = 'SkyLab'
                ref_label = 'reference PS sensitivity'

            figaxs[gamma][1][0].errorbar([cat_name], sens_dict['sens'], yerr=np.atleast_2d(sens_dict['sens_e']).T,
                                         color='red', label=fs_label, marker='o', capsize=3)

            if ps:
                figaxs[gamma][1][0].plot([cat_name], reference_sensitivity(sindec[cat_name], gamma=gamma),
                                         color='k', label=ref_label, marker='o')

            skylab_res = skylab_data(gamma, cat_name, northern=northern, weighted=weighted, ps=ps)
            logging.debug(f'skylab results: {skylab_res}')
            if not isinstance(skylab_res, type(None)):
                if ((skylab_res[2].decode()) != cat_name) or (float(skylab_res[3].decode()) != gamma):
                    raise Exception(f'Skylab results refers to SNe {skylab_res[2]} and gamma={skylab_res[3]} '
                                    f'not to SNe {cat_name} and gamma={gamma:.2f}')

                sres = float(skylab_res[0].decode())
                relative_to = sres if not ps else reference_sensitivity(sindec[cat_name], gamma)

                logging.debug(f'x={cat_name}, y={float(skylab_res[0].decode())}, yerr={float(skylab_res[1].decode())}')
                figaxs[gamma][1][0].errorbar([cat_name], sres, yerr=float(skylab_res[1].decode()),
                                             label=sl_label, color='blue', marker='o', capsize=3)
                figaxs[gamma][1][1].errorbar([cat_name], sres/relative_to,
                                             yerr=float(skylab_res[1].decode())/relative_to,
                                             color='blue', marker='', capsize=3)
                figaxs[gamma][1][1].errorbar([cat_name], sens_dict['sens']/relative_to,
                                             yerr=np.atleast_2d(sens_dict['sens_e']).T/relative_to,
                                             color='red', marker='', capsize=3)

            else:
                logging.info(f'No skylab results for SNe {cat_name} gamma={gamma}')

    # saving plots

    for gamma, figax in figaxs.items():

        figax[1][0].legend()
        figax[1][0].set_ylabel(r'flux [GeV$^{-1}$ s$^{-1}$ cm$^{-2}$]')
        figax[1][0].set_title('Sensitivity flux \n'
                              f'$\gamma$ = {gamma:.2f}')

        figax[1][1].axhline(1, color='k', ls='--', )
        figax[1][1].set_xlabel('SN type')
        figax[1][1].set_ylabel('ratio')
        figax[1][1].set_yscale('log')

        plt.tight_layout()

        fn = plot_output_dir(f'{raw}/gamma{gamma:.2f}_skylab_sn_cat_crosscheck.pdf')
        logging.debug(f'saving figure under {fn}')
        figax[0].savefig(fn)

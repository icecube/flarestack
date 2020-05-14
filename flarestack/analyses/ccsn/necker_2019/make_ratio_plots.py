from flarestack.analyses.ccsn.stasik_2017.shared_ccsn import limit_sens as stasik_sens_flarestack
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits, get_figure_limits, p_vals
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import limit_sens as necker_sens, raw_output_dir
from flarestack.analyses.ccsn import get_sn_color
from flarestack.shared import plot_output_dir
import pickle
import matplotlib.pyplot as plt
import numpy as np
import logging
import os


# define output directory
ratios_plot_dir = plot_output_dir(f'{raw_output_dir}/ratios')
if not os.path.isdir(ratios_plot_dir):
    logging.debug(f'making directory {ratios_plot_dir}')
    os.mkdir(ratios_plot_dir)

# use this minimizer
mh_name = 'fit_weights'

# spectral indice to be used
gamma = 2.

# base name
raw = f'{ratios_plot_dir}/{mh_name}'
if not os.path.isdir(raw):
    logging.debug(f'making directory {plot_output_dir(raw)}')
    os.mkdir(raw)

# loop over pdf types
for pdf_type in ['box', 'decay']:

    plot_dir = f'{raw}/{pdf_type}'
    if not os.path.isdir(plot_dir):
        logging.debug(f'making directory {plot_dir}')
        os.mkdir(plot_dir)

    # load saved sensitivities
    with open(necker_sens(mh_name, pdf_type), 'rb') as f:
        necker_sens_dict = pickle.load(f)

    # format nicer
    stacked_sens_necker = {
        cat_name: np.array(
            [(necker_sens_dict[cat_name][time_key][gamma][2], time_key)
             for time_key in necker_sens_dict[cat_name]], dtype='<f8'
        )
        for cat_name in necker_sens_dict
    }

    # load original results by Alex Stasik
    with open(stasik_sens_flarestack(mh_name, pdf_type), 'rb') as f:
        stasik_sens_flaresatck_dict = pickle.load(f)

    # format nicer
    stacked_sens_stasik_flaresatck = {
        cat_name: np.array(
            [(stasik_sens_flaresatck_dict[cat_name][time_key][gamma][2], time_key)
             for time_key in stasik_sens_flaresatck_dict[cat_name]], dtype='<f8'
        )
        for cat_name in stasik_sens_flaresatck_dict
    }

    # =========================     plot necker/stasik_flaresatck      ====================== #

    xlabel = 'Box function $\Delta T$ [d]' if 'box' in pdf_type else \
        't$_{\mathrm{decay}}$  [y]'

    fig, ax = plt.subplots()

    for cat_name in stacked_sens_necker:

        # res_stasik = get_figure_limits(cat_name, pdf_type)
        plot_arr = np.array(stacked_sens_necker[cat_name], dtype='<f8')
        stasik_arr = np.array(stacked_sens_stasik_flaresatck[cat_name if not 'P' in cat_name else 'IIp'], dtype='<f8')

        plot_arr[:, 0][np.argsort(plot_arr[:, 1])] /= stasik_arr[:, 0][np.argsort(stasik_arr[:, 1])]  # res_stasik['E'][np.argsort(res_stasik['t'])]

        logging.debug(f'plot array: \n {plot_arr}')

        ax.plot(plot_arr[:, 1], plot_arr[:, 0], marker='d', color=get_sn_color(cat_name),
                    label=f'{cat_name} sensitivity', ls='')

    ax.axhline(1, ls='--', color='k', label=f'ratio=1')

    ax.set_ylabel('$E^{\\nu}_{tot, new} / E^{\\nu}_{tot, \, Stasik \, reproduced}$')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    ax.legend()

    plt.grid()
    plt.title(f'Stacked sensitivity ratio for $\gamma = {gamma}$ \n \"new / Stasik reproduced with Flaresatck\"')
    plt.tight_layout()

    fname = f'{plot_dir}/stacked_sensitivity_ratio_reproduced_gamma{gamma}_{pdf_type}.pdf'
    logging.debug(f'saving figure under {fname}')

    plt.savefig(fname)
    plt.close()

    # =========================     plot necker/stasik_original      ====================== #

    fig, ax = plt.subplots()

    for cat_name in stacked_sens_necker:

        res_stasik = get_figure_limits(cat_name if not 'P' in cat_name else 'IIp', pdf_type)
        plot_arr = np.array(stacked_sens_necker[cat_name], dtype='<f8')

        plot_arr[:, 0][np.argsort(plot_arr[:, 1])] /= res_stasik['E'][np.argsort(res_stasik['t'])]

        logging.debug(f'plot array: \n {plot_arr}')

        # yerr = plot_arr[:, 0] * 0.1
        pval_mask = p_vals[pdf_type][cat_name if not 'P' in cat_name else 'IIp']['pval'] >= 0.5
        sens = plot_arr[pval_mask]
        no_sens = plot_arr[np.invert(pval_mask)]

        for arr, marker, label in zip(
                [sens, no_sens],
                ['d', 'x'],
                ['sensitivity', 'limit']
        ):
            ax.plot(arr[:, 1], arr[:, 0], marker=marker, color=get_sn_color(cat_name),
                    label=f'{cat_name} {label}', ls='')

    ax.axhline(1, ls='--', color='k', label=f'ratio=1')

    ax.set_ylabel('$E^{\\nu}_{tot, new} / E^{\\nu}_{tot, \, Stasik \, original}$')
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    ax.legend()

    plt.grid()
    plt.title(f'Stacked sensitivity ratio for $\gamma = {gamma}$ \n \"new / Stasik original\"')
    plt.tight_layout()

    fname = f'{plot_dir}/stacked_sensitivity_ratio_original_gamma{gamma}_{pdf_type}.pdf'
    logging.debug(f'saving figure under {fname}')

    plt.savefig(fname)
    plt.close()

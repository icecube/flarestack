"""Script to calculate and plot the sensitivity flux from the population of the different SNe types"""
import numpy as np
from astropy import units as u
from scipy.integrate import quad
from flarestack.shared import plot_output_dir
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import raw_output_dir, limit_sens, sn_times, \
    get_population_flux, pdf_names
from flarestack.analyses.ccsn.necker_2019.ccsn_rate import get_ccsn_rate_type_function
from flarestack.analyses.ccsn import get_sn_color
from flarestack.cosmo.neutrino_cosmology import get_diffuse_flux_at_1GeV
from flarestack.cosmo import get_diffuse_flux_contour
from flarestack.cosmo.icecube_diffuse_flux.joint_15 import contour_95, e_range, units
# from flarestack.misc.convert_diffuse_flux_contour import contour_95, lower_contour, upper_contour, e_range
import pickle
import matplotlib.pyplot as plt
import os
import logging


mh_name = 'fit_weights'
energy_range_gev = np.linspace(10**3, 10**7, 10)
linestyles = ['--', '-.', ':', '-']


plot_out_dir = plot_output_dir(f'{raw_output_dir}/sensitivity_fluxes/{mh_name}')
if not os.path.isdir(plot_out_dir):
    os.makedirs(plot_out_dir)

best_f, upper_f, lower_f, e_range = get_diffuse_flux_contour(contour_name=95)


def plot_diffuse_flux_measurement(axes, **kwargs):
    logging.debug('plotting diffuse flux measurement')
    label = kwargs.get('label', 'measured diffuse flux, 95% contour')
    patch = axes.fill_between(e_range,
                              y1=lower_f(e_range) * e_range ** 2,
                              y2=upper_f(e_range) * e_range ** 2,
                              label=label, color='k', alpha=0.5)
    return axes, patch


def get_stacked_sens_dict(pdf_type):
    base = raw_output_dir + f"/calculate_sensitivity_ps-v002p03/{mh_name}/{pdf_type}/"
    file = limit_sens(base)
    logging.debug(f'loading {file}')
    with open(file, 'rb') as f:
        stacked_sens_flux_dictionary = pickle.load(f)
    return stacked_sens_flux_dictionary


def add_population_flux(axis, sn_type, pdf_type, gamma_dict, gamma, time_key, **kwargs):

    sens, sens_e, energy = gamma_dict.get(gamma, [np.nan, np.nan, np.nan])
    if np.isnan(sens):
        logging.warning(f'No sensitivity for {sn_type}, {pdf_type}')

    z = kwargs.get('z', 2.5)
    pop_sens_flux = get_population_flux(energy * u.erg, get_ccsn_rate_type_function(sn_type), gamma, z)
    logging.debug(pop_sens_flux)

    default_label = f'{float(time_key) / 364.25} yr {pdf_type}' if pdf_type == 'decay' else \
        f'{time_key} day {pdf_type}'

    perc = pop_sens_flux/get_diffuse_flux_at_1GeV()[0]
    logging.debug(f'percentage of diffuse flux: {perc.value * 100:.2f}%')
    default_label += f': {perc.value * 100:.1f}%'

    sens_flux_time_esquared = pop_sens_flux * energy_range_gev ** (-1 * gamma) * \
                              energy_range_gev ** 2

    axis.plot(energy_range_gev, sens_flux_time_esquared,
              label=kwargs.get('label', default_label), ls=kwargs.get('ls', ''), color=get_sn_color(sn_type))

    return axis


def plot_sensitivity_populationflux_per_pdf_and_type(filename, gamma, **kwargs):

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(5.85*3/1.5, 3.6154988341868854*2/1.5),
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            sharex='all', sharey='all')
    diffuse_patch = None
    for i, (sn_type, pdf_dict) in enumerate(sn_times.items()):

        for j, pdf_type in enumerate(pdf_dict):
            stacked_sens_flux = get_stacked_sens_dict(pdf_type).get(sn_type, dict())
            axs[j][i].grid()

            for k, (time_key, time_dict) in enumerate(stacked_sens_flux.items()):
                axs[j][i] = add_population_flux(axs[j][i], sn_type, pdf_type, time_dict, gamma, time_key,
                                                ls=linestyles[k], **kwargs)

            axs[j][i], diffuse_patch = plot_diffuse_flux_measurement(axs[j][i], label='')

            axs[j][i].set_xscale('log')
            axs[j][i].set_yscale('log')
            axs[j][i].legend(loc='lower left')

        axs[0][i].set_title(sn_type)

    axs[0][-1].tick_params(labelbottom=True)
    axs[-1][-1].axis('off')  # switch off unused subplot

    xl, yl = axs[-1][-1].get_xlim(), axs[-1][-1].get_ylim()
    line_patch = axs[-1][-1].plot([0], [0], color='k', ls='-')
    axs[-1][-1].set_xlim(xl)
    axs[-1][-1].set_ylim(yl)

    fs = 11
    fig.legend([line_patch[0], diffuse_patch],
               ['sensitivity (this work)', 'IceCube diffuse flux,\n95% contour \nApJ 809, 2015'],
               loc='center', bbox_to_anchor=axs[-1][-1].get_position(), fontsize=fs)
    fig.text(0.5, 0.04, r"$E_{\nu}$ [GeV]", ha='center', fontsize=fs)  # xlabel
    fig.text(0.06, 0.5, r'$E^{2} \frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]',
             va='center', rotation='vertical', fontsize=fs)  # ylabel
    fig.suptitle(kwargs.get('title', ''), fontsize=fs)

    if kwargs.get('preliminary', True):
        empty_plot_position = axs[-1][-1].get_position().bounds
        icecube_preliminary_pos = (empty_plot_position[0] + empty_plot_position[2]/2,
                                   empty_plot_position[1])
        fig.text(*icecube_preliminary_pos, 'IceCube Preliminary',
                 color='r', fontsize=18, va='center', ha='center')

    logging.debug(f'saving figure under {filename}')
    fig.savefig(filename)
    plt.close()


def plot_sensitivity_flux_per_type(filename, gamma):


    plot_dict = dict()

    for i, (sn_type, pdf_dict) in enumerate(sn_times.items()):

        logging.debug(f'sntype is {sn_type}')
        plot_dict[sn_type] = dict()

        for j, pdf_type in enumerate(pdf_dict):
            logging.debug(f'pdf type {pdf_type}')
            stacked_sens_flux = get_stacked_sens_dict(pdf_type).get(sn_type, dict())

            for k, (time_key, time_dict) in enumerate(stacked_sens_flux.items()):

                logging.debug(f'time key {time_key}')
                t = float(time_key)/364.25 if 'decay' in pdf_type else float(time_key)
                pdf_name = pdf_names(pdf_type, t)
                logging.debug(f'pdf name {pdf_name}')
                res_list = time_dict.get(gamma, list())

                if len(res_list) < 1:
                    logging.warning(f'No sensitivity results for type {sn_type}, gamma {gamma}, {pdf_name}')

                else:
                    plot_dict[sn_type][pdf_name] = res_list

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5.85*2, 3.6154988341868854 + 0.1),
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            sharey='all')

    for i, (sn_type, pdf_dict) in enumerate(plot_dict.items()):
        logging.debug(f'plotting {sn_type} sensitivity')
        x = np.arange(len(pdf_dict.keys()))

        for pdf_key, res in pdf_dict.items():

            logging.debug(pdf_key)
            y = res[0]
            yerr = np.array([(res[1][0], res[1][1])]).T
            logging.debug(f'y={y}, yerr={yerr} with shape {np.shape(yerr)}')
            axs[i].errorbar([pdf_key], y, yerr=yerr, color=get_sn_color(sn_type), ls='', marker='o', ms='2',
                            capsize=5)
            axs[i].set_yscale('log')

        axs[i].set_xticks(x)
        axs[i].set_xticklabels(pdf_dict.keys(), rotation=45)
        axs[i].set_title(f'$\gamma$={gamma:.1f}\n\n{sn_type}' if i==1 else sn_type)
        axs[i].grid()

    axs[0].set_ylabel('Sensitivity flux @ 1GeV \n[GeV$^{-1}$ s$^{-1}$ cm$^{-2}$]')
    axs[1].set_xlabel('Time PDF')
    # fig.suptitle(rf'$\gamma$ = {gamma:.1f}')

    fig.tight_layout()
    logging.debug(f'saving under {filename}')
    fig.savefig(filename)
    plt.close()


def add_cumulative_flux_sntype(axis, gamma_dict, gamma, sn_type, rate_fct, zrange, **kwargs):

    rate_fct = get_ccsn_rate_type_function(sn_type)
    sens, sens_e, energy = gamma_dict.get(gamma, [np.nan, np.nan, np.nan])
    f = list()
    for z in zrange:
        f.append(get_population_flux(energy * u.erg, rate_fct, gamma, z))

    def integrand(zz): return get_population_flux(energy * u.erg, rate_fct, gamma, zz)
    norm_to = quad(integrand, 0, 2.3)[0]
    f = np.array(f) / norm_to

    axis.plot(zrange, f, label=kwargs.get('label', sn_type), color=kwargs.get('color', get_sn_color(sn_type)),
              **kwargs)

    return axis


def add_rate_sntype(axis, sn_type, zrange, normed=False, **kwargs):
    rate_fct = get_ccsn_rate_type_function(sn_type)
    rate = np.array([rate_fct(z) for z in zrange])
    norm_to = quad(rate_fct, 0, 2.3)[0] if normed else 1
    rate /= norm_to
    axis.plot(zrange, rate, color=kwargs.get('color', get_sn_color(sn_type)), label=kwargs.get('label', sn_type),
              **kwargs)
    return axis


def add_cumulative_rate_sntype(axis, sn_type, zrange, normed=False, **kwargs):
    rate_fct = get_ccsn_rate_type_function(sn_type)
    rate = [rate_fct(z) for z in zrange]
    cum_rate = np.cumsum(rate)
    norm_to = quad(rate_fct, 0, 2.3)[0] if normed else 1
    cum_rate /= norm_to
    axis.plot(zrange, cum_rate, color=kwargs.get('color', get_sn_color(sn_type)), label=kwargs.get('label', sn_type),
              **kwargs)
    return axis


def make_rate_plot(directory, gamma):

    zrange = np.linspace(0, 2.3, 100)

    rate_fig, rate_axs = plt.subplots(3, sharex='all', gridspec_kw={'hspace': 0})

    for i, sntype in enumerate(sn_times):

        rate_axs[i] = add_rate_sntype(rate_axs[i], sntype, zrange)
        rate_axs[i] = add_cumulative_rate_sntype(rate_axs[i], sntype, zrange)


if __name__ == '__main__':

    for energy_gamma in [2, 2.5]:

        this_plot_dir = f'{plot_out_dir}/{energy_gamma:.1f}'

        if not os.path.isdir(this_plot_dir):
            os.makedirs(this_plot_dir)

        fn = f'{this_plot_dir}/sensitivities_{energy_gamma:.1f}.pdf'
        plot_sensitivity_flux_per_type(fn, energy_gamma)

        title = 'Sensitivity Fluxes and Contribution to Diffuse Flux'

        fn = f'{this_plot_dir}/combined_{energy_gamma:.1f}_preliminary.pdf'
        plot_sensitivity_populationflux_per_pdf_and_type(fn, energy_gamma, z=2)

        fn = f'{this_plot_dir}/combined_{energy_gamma:.1f}.pdf'
        plot_sensitivity_populationflux_per_pdf_and_type(fn, energy_gamma, z=2, preliminary=False)

        fn = f'{this_plot_dir}/combined_{energy_gamma:.1f}_test_redshift8.pdf'
        plot_sensitivity_populationflux_per_pdf_and_type(fn, energy_gamma, z=8)



import os
import logging
from astropy import units as u
from flarestack.shared import limits_dir, limit_output_path
from flarestack.cosmo.neutrino_cosmology import define_cosmology_functions
from flarestack.core.energy_pdf import EnergyPDF


raw_output_dir = 'analyses/ccsn/necker_2019'

ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_necker_cat_dir = ccsn_cat_dir + "raw_necker/"
stasik_cat_dir = ccsn_cat_dir + 'raw_stasik/'

sn_cats = ["IIn", "IIP", "Ibc"]

sn_times_box = [100, 300, 1000]  # in days
sn_times_decay = [0.02, 0.2, 2]  # in years
sn_times_dict = {'box': sn_times_box, 'decay': sn_times_decay}

sn_times = {'IIn': sn_times_dict, 'IIP': sn_times_dict,
            'Ibc': {'box': sn_times_box + [-20]}}

conservative_redshift_addition = 0.001


def updated_sn_catalogue_name(sn_type, pdf_name='', flagged=False, z_conservative=conservative_redshift_addition):

    if pdf_name:
        pdf_name = '_' + pdf_name

    sn_name = sn_type + pdf_name

    if pdf_name or flagged:
        directory = raw_necker_cat_dir
    else:
        directory = ccsn_cat_dir

    if z_conservative != conservative_redshift_addition:
        directory += f'zplus{z_conservative}/'
        if not os.path.exists(directory):
            logging.debug(f'making directory {directory}')
            os.mkdir(directory)

    if flagged:
        directory += 'flagged/'
        if not os.path.exists(directory):
            logging.debug(f'making directory {directory}')
            os.mkdir(directory)

    return directory + sn_name + '.npy'


def raw_sn_catalogue_name(sn_type, person="necker", fs_readable=True):

    if person == 'necker':

        return raw_necker_cat_dir + sn_type + '.csv'

    elif person == 'stasik':

        if 'Ib' in sn_type:
            res =  stasik_cat_dir + 'Ib_BoxPre20.0_New'
        elif sn_type == 'IIn':
            res = stasik_cat_dir + 'IIn_Box300.0_New'
        elif sn_type in ('IIP', 'IIp'):
            res = stasik_cat_dir + 'IIp_Box300.0_New'
        else:
            raise ValueError(f'input for sn_type: {sn_type} not understood')

        if fs_readable:
            res += '_fs_readable'

        res += '.npy'

        return res

    else:

        raise ValueError(f'input for person: {person} not understood')


def sn_time_pdfs(sn_type, pdf_type='box'):

    time_pdfs = []

    # to ensure combatipility with stasik stuff where it's IIp and not IIP
    sn_type = sn_type if sn_type != 'IIp' else 'IIP'

    if pdf_type == 'box':

        for i in sn_times[sn_type][pdf_type]:

            pdf_dict = {
                "time_pdf_name": pdf_type,
                "pre_window": 0,
                "post_window": 0
            }

            if i > 0:
                pdf_dict['post_window'] = i
            elif i < 0:
                pdf_dict['pre_window'] = abs(i)
            else:
                raise ValueError(f'time for PDF type {pdf_type} has to be bigger or smaller than zero, not {i}')

            time_pdfs.append(pdf_dict)

    elif pdf_type == 'decay':

        for i in sn_times[sn_type][pdf_type]:

            pdf_dict = {
                'time_pdf_name': pdf_type,
                'decay_time': i * 364.25,  # convert to days
                'decay_length': (1000 - 1) * i * 364.25
            }

            time_pdfs.append(pdf_dict)

    else:
        raise ValueError(f'Input {pdf_type} for PDF type not understood!')

    return time_pdfs


def pdf_names(pdf_type, pdf_time):
    if pdf_time < 0: pdf_time = f'Pre{abs(pdf_time):.0f}'
    elif pdf_time == 2: pdf_time = f'{pdf_time:.0f}'
    elif abs(pdf_time) >= 100: pdf_time = f'{pdf_time:.0f}'
    return f'{pdf_type}{pdf_time}'


def limit_sens(base):

    # base = f'{raw_output_dir}/calculate_sensitivity/{mh_name}/{pdf_type}/'
    sub_base = base.split(os.sep)

    for i, _ in enumerate(sub_base):
        p = sub_base[0]
        for d in range(1, i):
            p += f'{os.sep}{sub_base[d]}'
        p = limits_dir + p
        if not os.path.isdir(p):
            logging.debug(f'making directory {p}')
            os.mkdir(p)

    return limit_output_path(base)


def get_cosmology(nu_e, rate, gamma):
    energy_pdf_dict = {
        'energy_pdf_name': 'power_law',
        'gamma': gamma,
    }
    energy_pdf = EnergyPDF.create(energy_pdf_dict)
    fluence_conversion = energy_pdf.fluence_integral() * u.GeV ** 2
    nu_e = nu_e.to("GeV") / fluence_conversion
    functions = define_cosmology_functions(rate, nu_e, gamma)
    return functions


def get_population_flux(nu_e, rate, gamma, redshift):
    functions = get_cosmology(nu_e, rate, gamma)
    cumulative_flux = functions[3]
    return cumulative_flux(redshift)[-1]

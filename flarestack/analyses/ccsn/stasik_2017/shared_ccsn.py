from __future__ import print_function
import os
import logging
import numpy as np
import pickle as Pickle
from scipy.interpolate import interp1d
from flarestack.shared import limit_output_path, limits_dir
from astropy import units as u
from astropy.table import Table

ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_cat_dir = ccsn_cat_dir + "raw/"

sn_cats = ["IIn", "IIp", "Ibc"]
sn_times_box = [100, 300, 1000]
sn_times_decay = [0.02, 0.2, 2]
sn_times_dict = {'box': sn_times_box, 'decay': sn_times_decay}

sn_times = {'IIn': sn_times_dict, 'IIP': sn_times_dict,
            'Ibc': {'box': sn_times_box + [-20]}}


def raw_sn_catalogue_name(sn_type):
    return f"{raw_cat_dir}/{sn_type}_original.csv"


def pdf_names(pdf_type, pdf_time):

    logging.debug(f'getting pdf name for type {pdf_type} {pdf_time}')

    if pdf_time < 0:
        pdf_time_str = f'Pre{abs(pdf_time):.0f}'
    elif pdf_time < 1:
        pdf_time_str = f'{pdf_time}'
    elif pdf_time < 100:
        pdf_time_str = f'{pdf_time:.1f}'
    else:
        pdf_time_str = f'{pdf_time:.0f}'

    return f'{pdf_type}{pdf_time_str}'


def sn_catalogue_name(sn_type, nearby=True, raw=False, pdf_name=''):

    pdf_name = None if pdf_name == '' else pdf_name
    if sn_type == 'IIp': sn_type = 'IIP'

    if raw:
        sn_name = 'raw/'

        if 'Ib' in sn_type:
            sn_name += 'Ib_BoxPre20.0'
        elif 'IIn' in sn_type:
            sn_name += 'IIn_Box300.0'
        elif ('IIp' in sn_type) or ('IIP' in sn_type):
            sn_name += 'IIp_Box300.0'
        else:
            raise Exception

        sn_name += '_New_fs_readable.npy'

        return ccsn_cat_dir + sn_name

    else:

        sn_name = sn_type
        if pdf_name: sn_name += '_' + pdf_name
        if nearby: sn_name += '_nearby'
        sn_name += '.npy'

        # if nearby:
        #     sn_name += "nearby.npy"
        # else:
        #     sn_name += "distant.npy"

        if pdf_name:
            res = raw_cat_dir + sn_name
        else:
            res = ccsn_cat_dir + sn_name

        return res


def show_cat(*args, **kwargs):

    file = sn_catalogue_name(*args, **kwargs)
    tab = Table(np.load(file))
    print(tab)
    return tab


def sn_time_pdfs(sn_type):

    time_pdfs = []

    for i in sn_times:
        time_pdfs.append(
            {
                "time_pdf_name": "box",
                "pre_window": 0,
                "post_window": i
            }
        )

    if sn_type == "Ibc":
        time_pdfs.append(
            {
                "time_pdf_name": "box",
                "pre_window": 20,
                "post_window": 0
            }
        )

    return time_pdfs


def limit_sens(mh_name, pdf_type):

    base = f'analyses/ccsn/stasik2017/calculate_sensitivity/{mh_name}/{pdf_type}/'
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


def ccsn_limits(sn_type):

    base = "analyses/ccsn/stasik_2017/calculate_sensitivity/"
    path = base + sn_type + "/real_unblind/"

    savepath = limit_output_path(path)

    print("Loading limits from", savepath)
    with open(savepath, "r") as f:
        results = Pickle.load(f)
    return results


def ccsn_energy_limit(sn_type, gamma):
    results = ccsn_limits(sn_type)

    spline_y = np.exp(interp1d(results["x"], np.log(results["energy"]))(gamma))

    return spline_y * u.erg

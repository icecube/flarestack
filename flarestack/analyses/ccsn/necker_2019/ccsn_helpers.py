import os
# from flarestack.analyses.ccsn.necker2019.ccsn_helpers import sn_catalogue_name

raw_output_dir = 'analyses/ccsn/necker_2019'

ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_necker_cat_dir = ccsn_cat_dir + "raw_necker/"
stasik_cat_dir = ccsn_cat_dir + 'raw_stasik/'

sn_cats = ["IIn", "IIP", "Ibc"]

sn_times_box = [100, 300, 1000]
sn_times_decay = [0.02, 0.2, 2]
sn_times_dict = {'box': sn_times_box, 'decay': sn_times_decay}

sn_times = {'IIn': sn_times_dict, 'IIP': sn_times_dict,
            'Ibc': {'box': sn_times_box + [-20]}}


def updated_sn_catalogue_name(sn_type, pdf_name='', flagged=False, nearby=True):
    if pdf_name: pdf_name = '_' + pdf_name
    sn_name = sn_type + pdf_name + '.npy'
    if flagged: sn_name = 'flagged/' + sn_name

    if pdf_name or flagged:
        return raw_necker_cat_dir + sn_name
    else:
        return ccsn_cat_dir + sn_name


def raw_sn_catalogue_name(sn_type, person="necker"):

    if person == 'necker':

        return raw_necker_cat_dir + sn_type + '.csv'

    elif person == 'stasik':

        if 'Ib' in sn_type:
            return stasik_cat_dir + 'Ib_BoxPre20.0_New.npy'
        elif sn_type == 'IIn':
            return stasik_cat_dir + 'IIn_Box300.0_New.npy'
        elif sn_type in ('IIP', 'IIp'):
            return stasik_cat_dir + 'IIp_Box300.0_New.npy'
        else:
            raise ValueError(f'input for sn_type: {sn_type} not understood')

    else:

        raise ValueError(f'input for person: {person} not understood')


def sn_time_pdfs(sn_type, pdf_type='box'):

    # TODO: implement decay function

    time_pdfs = []

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
        raise NotImplementedError

    else:
        raise ValueError(f'Input {pdf_type} for PDF type not understood!')

    return time_pdfs


def pdf_names(pdf_type, pdf_time):
    if pdf_time < 0: pdf_time = f'Pre{abs(pdf_time)}'
    return f'{pdf_type}{pdf_time}'

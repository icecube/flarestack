import os
# from flarestack.analyses.ccsn.necker2019.ccsn_helpers import sn_catalogue_name


ccsn_dir = os.path.abspath(os.path.dirname(__file__))
ccsn_cat_dir = ccsn_dir + "/catalogues/"
raw_necker_cat_dir = ccsn_cat_dir + "raw_necker/"
stasik_cat_dir = ccsn_cat_dir + 'raw_stasik/'

sn_cats = ["IIn", "IIp", "Ibc"]

sn_times_box = [100, 300, 1000]
sn_times_decay = [0.02, 0.2, 2]
sn_times_dict = {'box': sn_times_box, 'decay': sn_times_decay}

sn_times = {'IIn': sn_times_dict, 'IIP': sn_times_dict,
            'Ibc': {'box': sn_times_box + [-20]}}


def updated_sn_catalogue_name(sn_type, pdf_name, nearby=True):
    sn_name = sn_type + "_" + pdf_name + '.npy'

    # if nearby:
    #     sn_name += "nearby.npy"
    # else:
    #     sn_name += "distant.npy"

    return ccsn_cat_dir + sn_name


def raw_sn_catalogue_name(sn_type, person = "necker"):

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


def sn_time_pdfs(sn_type, pdf_type = 'box'):

    # TODO: implement decay function

    time_pdfs = []

    for i in sn_times[sn_type][pdf_type]:
        time_pdfs.append(
            {
                "time_pdf_name": "box",
                "pre_window": 0,
                "post_window": i
            }
        )

    return time_pdfs


def pdf_names(pdf_type, pdf_time):
    if pdf_time < 0: pdf_time = f'Pre{abs(pdf_time)}'
    return f'{pdf_type}{pdf_time}'


# def pdf_names_for_filenames(pdf_name):
#
#     if "box" in pdf_name: kind = 'box'
#     elif 'decay' in pdf_name: kind = 'decay'
#     else: raise ValueError(f'kind of PDF {pdf_name} not know')
#
#

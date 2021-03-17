import pandas as pd
import numpy as np
import logging
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import \
    raw_sn_catalogue_name, updated_sn_catalogue_name, sn_times, pdf_names, conservative_redshift_addition
from flarestack.analyses.ccsn.stasik_2017.shared_ccsn import sn_catalogue_name
import math


logging.getLogger().setLevel("INFO")

# columns in raw catalogues
columns = np.array(['name', 'discovery', 'ra', 'dec', 'redshift', 'lum dist [Mpc]', 'weight', 'flag'])

# columns for output catalogues
columns_out = columns.copy()
columns_out[columns == 'lum dist [Mpc]'] = 'distance'
columns_out = np.array([
     'source_name',
     'ref_time_mjd',
     'ra_rad',
     'dec_rad',
     'redshift',
     'distance_mpc',
     'weight',
     'flag'
])

# columns to be added
to_add = {
    'names': ['injection_weight_modifier'],
    'values': [1.],
    'formats': ['<f8']
}

columns_out = np.append(columns_out, to_add['names'])

# data type for PDF catalogues
dt = {
    'names': columns_out,
    'formats': ['<U50', 'i'] + ['<f8'] * 6 + to_add['formats']
}

# columns that are not to be replaced when looking for values in catalogues of different PDFs
not_to_be_replaced = ['weight']
not_to_be_replaced_inds = [np.where(columns == col_name)[0][0] for col_name in not_to_be_replaced]
keep_inds = np.delete(np.array(range(len(columns))), not_to_be_replaced_inds)
keep_inds_out = np.append(keep_inds, [np.where(columns_out == col)[0][0] for col in to_add['names']])

# data type for combined catalogues
dt_comb = {
    'names': columns_out[keep_inds_out],
    'formats': np.array(dt['formats'])[keep_inds_out]
}


def load_catalogue(sn_type, pdf_name, include_flagged=False, z_add=0):
    """
    load the catalogue from the csv file
    :param sn_type: string, e.g. 'IIP'
    :param pdf_name: string, e.g. '300 day box', '0.2 decay function', 'missed_objects'
    :param include_flagged: bool, if True also objects that are flagged are included
    :param z_add: float, optional, if given, this number is added to all redshifts
    :return: np.ndarray, columns name, discovery, ra, dec, redshift, distance, weight
    """

    logging.info(f'loading the catalogue for {sn_type}: {pdf_name}')
    csv_filename = raw_sn_catalogue_name(sn_type)

    logging.info(f'filename: {csv_filename}')
    raw_catalogues = pd.read_csv(csv_filename)

    # get only the part of the catalogue that belongs to the given PDF
    col_catalogue = np.array(raw_catalogues.catalogue)
    logging.debug(f'looking for the right pdf in {col_catalogue}')
    pdf_names_mask = [type(ob) is str for ob in col_catalogue]
    pdf_names = col_catalogue[pdf_names_mask]
    pdf_names_inds = np.where(pdf_names_mask)[0]

    if len(np.where(pdf_names == pdf_name)[0]) < 1:
        raise IndexError(f'PDF {pdf_name} not available for {sn_type}')
    else:
        pdf_name_ind = np.where(pdf_names == pdf_name)[0][0]

    # if the last PDF in the list is selected, get everything until last row
    if pdf_name_ind == len(pdf_names) - 1:
        inds = [pdf_names_inds[pdf_name_ind]+3, None]
    else:
        inds = [pdf_names_inds[pdf_name_ind]+3, pdf_names_inds[pdf_name_ind+1]-2]

    logging.debug(f'The pdf names are stored under indices {pdf_names_inds}')
    logging.debug(f'Getting elements [{inds[0]} : {inds[1]}]')

    name_ind_in_columns = np.where(columns == 'name')[0][0]
    logging.debug(f'index of "name" in columns is {name_ind_in_columns}')

    catalogue = np.array(
        raw_catalogues.values[
                         inds[0] : inds[1],
                         keymap(columns, raw_catalogues)
                         ]
    )

    logging.debug(f'after selecting right PDF: {catalogue}')

    # get only rows with the values, e.g. the rows where 'name' is not NaN
    objects_names_inds = []

    for i, element in enumerate(catalogue[:, name_ind_in_columns]):

        if type(element) is (list or np.array):

            if len(element) > 1:
                raise ValueError(f'more than one name for object')
            else: element = element[0]

        if type(element) is str: objects_names_inds += [i]

    logging.debug(f'getting the rows with indices {objects_names_inds}')

    catalogue = catalogue[objects_names_inds, :]
    logging.debug(f'after removing NaNs from the name column: \n {catalogue}')

    # for empty rows, get the values from other catalogues (except weights!)
    for i, row in enumerate(catalogue):

        this_discovery = row[columns == 'discovery'][0]

        if np.isnan(this_discovery):
            logging.debug(f'discovery for row {i} is NaN')

            raw_cat_arr = raw_catalogues.values[:, keymap(columns, raw_catalogues)]

            # select only rows where the name is the name of this row
            raw_cat_arr = raw_cat_arr[raw_cat_arr[:, name_ind_in_columns] == row[name_ind_in_columns], :]

            replacement = raw_cat_arr[
                np.invert(
                    np.isnan(
                        np.array(raw_cat_arr[:, np.where(columns == 'discovery')[0][0]],
                                 dtype='<f8')
                    )
                ),
                keep_inds
            ]

            logging.debug(f'replacing this row with {replacement}')

            catalogue[i][keep_inds] = replacement

    logging.debug(f'after replacing missing fields with values from other PDFs: \n {catalogue}')

    add_list = [to_add['values']] * len(catalogue)
    logging.debug(f'adding {add_list} to array')

    catalogue = np.append(catalogue, add_list, axis=1)

    logging.debug(f'after adding columns: \n {catalogue}')

    # convert strings with commas to floats
    for i, format in enumerate(dt['formats']):

        if ('f' in format) and (type(catalogue[0,i]) is str):

            logging.debug(f'converting {columns_out[i]}: {catalogue[:,i]} to floats')
            catalogue[:,i] = [float(st.replace(',', '.')) for st in catalogue[:, i]]

    logging.debug(f'after converting strings to floats: \n {catalogue}')

    # set the right data type to array
    catalogue = np.array(
        [tuple(row) for row in catalogue],
        dtype=dt
    )

    logging.debug(f'after setting the right data type: \n {catalogue}')

    # convert ra and dec to radians
    conversion_factor = math.pi/180
    catalogue['ra_rad'] = catalogue['ra_rad']*conversion_factor
    catalogue['dec_rad'] = catalogue['dec_rad']*conversion_factor

    logging.debug(f'after converting to radians: \n {catalogue}')

    if not include_flagged:
        catalogue = catalogue[np.isnan(catalogue['flag'])]
        logging.debug(f'after removing flagged objects: \n {catalogue}')

    if z_add:
        logging.debug(f'adding {z_add} to all redshifts!')
        catalogue['redshift'] += z_add

    return catalogue


def keymap(keys, raw_catalogue):

    indices = [
        np.where(np.array(raw_catalogue.columns) == key)[0][0]
        if len(np.where(np.array(raw_catalogue.columns) == key)) <= 1
        else np.where(np.array(raw_catalogue.columns) == key)
        for key in keys
    ]

    logging.debug(f'keymap gives indices {indices} for keys {keys}')
    return indices


class InconsistencyError(Exception):
    def __init__(self, msg):
        self.message = msg


# ======================== #
# === execute building === #
# ======================== #

if __name__ == '__main__':

    logging.info('building catalogues')

    for z_add in [0, conservative_redshift_addition]:
        if z_add:
            logging.debug(f'adding {z_add} to all redshifts!')

        for flag in [True, False]:
            msg = 'Including flagged objects' if flag else 'Not including flagged objects'
            logging.info(msg)

            for sn_type in sn_times:
                logging.info(f'building catalogue for sn type {sn_type}')
                start = True

                for pdf_type in sn_times[sn_type]:
                    logging.info(f'pdf type: {pdf_type}')

                    for pdf_time in sn_times[sn_type][pdf_type]:

                        # get catalogues for individual PDFs
                        pdf_name = pdf_names(pdf_type, pdf_time)
                        logging.info(f'pdf time: {pdf_time}')
                        catalogue = load_catalogue(sn_type, pdf_name, include_flagged=flag, z_add=z_add)
                        savename = updated_sn_catalogue_name(sn_type, pdf_name, flagged=flag, z_conservative=z_add)
                        np.save(savename, catalogue)

                        # combine with previous PDF-catalogues
                        catalogue_red = catalogue[columns_out[keep_inds_out]]
                        if start: combined_catalogue = catalogue_red
                        start = False

                        new_mask = np.invert([name in combined_catalogue['source_name']
                                              for name in catalogue_red['source_name']])
                        new_objects = catalogue_red[new_mask]
                        combined_catalogue = np.array(list(combined_catalogue) + list(new_objects),
                                                      dtype=dt_comb)
                        # check consistency
                        # treat field 'flag' separately because difficulty with comparing NaNs in lists
                        special_field = 'flag'
                        to_check = columns_out[keep_inds_out][np.invert(columns_out[keep_inds_out] == special_field)]

                        for old in catalogue_red:

                            element_in_combined_catalogue = \
                                combined_catalogue[combined_catalogue['source_name'] == old['source_name']][0]

                            if (not np.array_equiv(old[to_check] ,element_in_combined_catalogue[to_check])) or \
                                    not (
                                            np.isnan(old[special_field]) is
                                            np.isnan(element_in_combined_catalogue[special_field])
                                    ):

                                logging.debug(f"old: type of flag is {type(old['flag'])}")
                                logging.debug(f"combined: type of flag is {type(element_in_combined_catalogue['flag'])}")
                                raise InconsistencyError(
                                    f'Inconsistency found in catalogues: \n '
                                    f'this catalogue {old} \n '
                                    f'combined {element_in_combined_catalogue}'
                                )

                if not flag:
                    # add previously missed objects
                    logging.info('adding previously missed objects')
                    previously_missed = load_catalogue(sn_type, 'missed_objects')
                    savename = updated_sn_catalogue_name(sn_type, 'missed_objects', z_conservative=z_add)
                    np.save(savename, previously_missed)

                    to_add_to_cat = previously_missed[columns_out[keep_inds_out]]
                    logging.debug(f'adding these previously missed ones: \n {to_add_to_cat}')
                    combined_catalogue = np.array(list(combined_catalogue) + list(to_add_to_cat), dtype=dt_comb)

                sname_combined = updated_sn_catalogue_name(sn_type, flagged=flag, z_conservative=z_add)
                logging.info(f'saving the combined catalogue for {sn_type} {msg} to {sname_combined}')
                np.save(sname_combined, combined_catalogue)

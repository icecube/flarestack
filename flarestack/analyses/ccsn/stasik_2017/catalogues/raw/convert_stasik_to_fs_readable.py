import numpy as np
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import (
    raw_sn_catalogue_name,
    sn_cats,
)


def convert_stasik(sn_type):
    path_raw = raw_sn_catalogue_name(sn_type, person="stasik", fs_readable=False)
    path_new = raw_sn_catalogue_name(sn_type, person="stasik", fs_readable=True)

    raw_cat = np.load(path_raw)
    dtype_names_raw = list(raw_cat.dtype.names)
    dtype_fields_raw = raw_cat.dtype.fields

    dtype_new = []

    for element_name in dtype_names_raw:
        element_dtype = dtype_fields_raw[element_name][0]

        if ("ra" in element_name) or ("dec" in element_name):
            new_element_name = element_name + "_rad"
        elif "distance" in element_name:
            new_element_name = element_name + "_mpc"
        elif "discoverydate_mjd" in element_name:
            new_element_name = "ref_time_mjd"
        elif element_name == "name":
            new_element_name = "source_name"
        else:
            new_element_name = element_name

        dtype_new.append((new_element_name, element_dtype))

    dtype_new.append(("injection_weight_modifier", "<f8"))

    # adding the value 1 for the column 'injection_weight_modifier'
    new_list = [tuple(list(row) + [1.0]) for row in raw_cat]

    new_cat = np.array(new_list, dtype=dtype_new)
    np.save(path_new, new_cat)


if __name__ == "__main__":
    for sn_type in sn_cats:
        convert_stasik(sn_type)

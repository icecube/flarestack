"""
This module provides the basic data types used by the other modules of flarestack.
"""

import numpy as np

""" Catalogue data type """
catalogue_dtype = [
    ("ra_rad", np.float),
    ("dec_rad", np.float),
    ("base_weight", np.float),
    ("injection_weight_modifier", np.float),
    ("ref_time_mjd", np.float),
    ("start_time_mjd", np.float),
    ("end_time_mjd", np.float),
    ("distance_mpc", np.float),
    ("source_name", "a30"),
]

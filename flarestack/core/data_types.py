"""
This module provides the basic data types used by the other modules of flarestack.
"""

import numpy as np

""" Catalogue data type """
catalogue_dtype = [
    ("ra_rad", float),
    ("dec_rad", float),
    ("base_weight", float),
    ("injection_weight_modifier", float),
    ("ref_time_mjd", float),
    ("start_time_mjd", float),
    ("end_time_mjd", float),
    ("distance_mpc", float),
    ("source_name", "a30"),
]

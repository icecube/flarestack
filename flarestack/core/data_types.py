import numpy as np

""" Catalogue data type """
cat_dtype = [
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

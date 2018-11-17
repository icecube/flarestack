from flarestack.utils.prepare_catalogue import custom_sources
from astropy.coordinates import Distance
from flarestack.shared import catalogue_dir
from flarestack.analyses.tde.shared_TDE import individual_tde_cat
import numpy as np
import os

# Max time taken from https://arxiv.org/pdf/1807.05965
# Corresponding time window set by method laid out in TDE analysis page:
#   https://wiki.icecube.wisc.edu/index.php/TDE

ref_time = 58286.9

t_start = ref_time - 30.
t_end = ref_time + 100.

# Ra and dec of source, from TNS (https://wis-tns.weizmann.ac.il/object/2018cow)
ra = 244.000927647
dec = 22.2680094118

# Distance to source,from http://www.astronomerstelegram.org/?read=11727
z = 0.014145
lumdist = Distance(z=z).to("Mpc").value

# Creates the .npy source catalogue
at2018cow_catalogue = custom_sources(
    name="AT2018cow",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=lumdist,
    start_time=t_start,
    end_time=t_end,
    ref_time=t_start
)

at2018cow_cat_path = individual_tde_cat("AT2018cow")

try:
    os.makedirs(os.path.dirname(at2018cow_cat_path))
except OSError:
    pass

np.save(at2018cow_cat_path, at2018cow_catalogue)

from flarestack.utils.prepare_catalogue import custom_sources
from astropy.coordinates import Distance
from flarestack.shared import transients_dir
import numpy as np

# Start and end time of neutrino flare, taken from box fit in
# https://arxiv.org/abs/1807.08794.

t_start = 56937.81
t_end = 57096.21

# Ra and dec of source, from Science paper (https://arxiv.org/abs/1807.08794)
ra = 77.3582
dec = 5.69314

# Distance to source, according to https://arxiv.org/abs/1802.01939, is 0.3365
z = 0.3365
lumdist = Distance(z=z).to("Mpc").value

# Creates the .npy source catalogue
txs_catalogue = custom_sources(
    name="TXS_0506+056",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=lumdist,
    start_time=t_start,
    end_time=t_end,
    ref_time=t_start
)

txs_cat_path = transients_dir + "TXS_0506+056.npy"
np.save(txs_cat_path, txs_catalogue)

from flarestack.utils.prepare_catalogue import custom_sources
from astropy.coordinates import Distance
from flarestack.shared import transients_dir

# A description of the source can be found on tevcat, with ra/dec and redshift
# http://tevcat.uchicago.edu/?mode=1;id=79

# Start and end time of flare in MJD
t_start = 57506.00
t_end = 57595.00

# Ra and dec of source
ra = 300.00
dec = 65.15

# Distance to source, according to https://arxiv.org/abs/1802.01939, is 0.3365
z = 0.3365
lumdist = Distance(z=z).to("Mpc").value

# Creates the .npy source catalogue
catalogue = custom_sources(
    name="TXS_0596+56",
    ra=ra,
    dec=dec,
    weight=1.,
    distance=lumdist,
    start_time=t_start,
    end_time=t_end,
)

cat_path = transients_dir + "TXS_0506+56.npy"
np.save(cat_path, catalogue)
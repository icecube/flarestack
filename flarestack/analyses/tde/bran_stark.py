from astropy import units as u
from flarestack.utils.neutrino_cosmology import integrate_over_z, define_cosmology_functions

def uniform_evolution(z):
    return 1.

def tde_rate(z):
    """TDE rate as a function of redshift, equal to the local density
    multiplied by the redshift evolution of the density. Rate is derived
    from https://arxiv.org/pdf/1707.03458

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return (uniform_evolution(z) * 8 * 10**-7) / (u.Mpc**3 * u.yr)

rate_per_z, _, _, _ = define_cosmology_functions(tde_rate, 1., 1.)

int_count = integrate_over_z(rate_per_z, 0.0, 0.1)

# Northern
hem_eff = 0.5

# Chip Gaps
cg_eff = 0.9

# Sun
sun_eff = 2./3.

eff = (hem_eff * cg_eff)

print(eff*int_count)


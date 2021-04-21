import logging
from astropy import units as u
from flarestack.cosmo.rates.sfr_rates import get_sfr_evolution

local_fbot_rates = {
    "ho_20_high": (7. * 10**-7./(u.Mpc**3 * u.yr), "https://arxiv.org/abs/2003.01222"),
    "ho_20_low": (4. * 10**-7./(u.Mpc**3 * u.yr), "https://arxiv.org/abs/2003.01222"),
}


def get_local_fbot_rate(rate_name=None, with_range=False):
    """Returns a local rate of Fast Blue Optical Transients (FBOTs).

    :param rate_name: Name of local FBOT rate to be used
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Local rate
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "ho_20_high"

    if rate_name not in local_fbot_rates.keys():
        raise Exception(f"Rate name '{rate_name}' not recognised. "
                        f"The following source evolutions are available: {local_fbot_rates.keys()}")
    else:
        local_rate, ref = local_fbot_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    if with_range:
        raise Exception(f"No one sigma rate range found for rate '{rate_name}'. "
                        f"Use a different rate, or set 'with_range=False'.")

    return local_rate.to("Mpc-3 yr-1")


def get_fbot_rate(evolution_name=None, rate_name=None, with_range=False, **kwargs):
    """Returns a local rate of Fast Blue Optical Transients (FBOTs) as a function of redshift.

    :param evolution_name: Name of Star Formation evolution to use
    :param rate_name: Name of local FBOT rate to be used
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Rate as a function of redshift
    """

    normed_evolution = get_sfr_evolution(evolution_name=evolution_name, **kwargs)
    local_rate = get_local_fbot_rate(rate_name=rate_name, with_range=with_range)

    return lambda z: normed_evolution(z) * local_rate

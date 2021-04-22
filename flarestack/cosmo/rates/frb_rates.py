import logging
from astropy import units as u
from flarestack.cosmo.rates.sfr_rates import get_sfr_evolution

local_frb_rates = {
    "bochenek_20": (
        7.23 * 10**7. / (u.Gpc**3 * u.yr),
        (7.23 - 6.13) * 10**7. / (u.Gpc**3 * u.yr),
        (7.23 + 8.78) * 10**7. / (u.Gpc**3 * u.yr),
        "https://arxiv.org/abs/2005.10828"
    ),
}


def get_local_frb_rate(rate_name=None, with_range=False):
    """Returns a local rate of Fast Radio Bursts (FBBs).

    :param rate_name: Name of local FRB rate to be used
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Local rate
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "bochenek_20"

    if rate_name not in local_frb_rates.keys():
        raise Exception(f"Rate name '{rate_name}' not recognised. "
                        f"The following rates are available: {local_frb_rates.keys()}")
    else:

        local_rate, lower_lim, upper_lim, ref = local_frb_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    if with_range:

        if lower_lim is None:
            raise Exception(f"No one sigma rate range found for rate '{rate_name}'. "
                            f"Use a different rate, or set 'with_range=False'.")

        return local_rate.to("Mpc-3 yr-1"), lower_lim.to("Mpc-3 yr-1"), upper_lim.to("Mpc-3 yr-1")

    else:
        return local_rate.to("Mpc-3 yr-1")


def get_frb_rate(evolution_name=None, rate_name=None, with_range=False, **kwargs):
    """Returns a local rate of core-collapse supernovae (CCSNe) as a function of redshift.

    :param evolution_name: Name of Star Formation evolution to use
    :param rate_name: Name of local FRB rate to be used
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Rate as a function of redshift
    """

    normed_evolution = get_sfr_evolution(evolution_name=evolution_name, **kwargs)
    local_rate = get_local_frb_rate(rate_name=rate_name, with_range=with_range)

    if with_range:
        return lambda z: local_rate[0] * normed_evolution(z), \
               lambda z: local_rate[1] * normed_evolution(z), \
               lambda z: local_rate[2] * normed_evolution(z)
    else:
        return lambda z: local_rate*normed_evolution(z)

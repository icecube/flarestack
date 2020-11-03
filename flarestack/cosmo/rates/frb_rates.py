import logging
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from flarestack.cosmo.rates.sfr_rates import get_sfr_evolution

local_frb_rates = {
    "bochenek_20": (7.23  * 10**7.  / (u.Gpc**3 * u.yr), "https://arxiv.org/abs/2005.10828"),
}

def get_local_frb_rate(rate_name=None):
    """Returns a local rate of Fast Radio Bursts (FBBs).

    :param rate_name: Name of local FRB rate to be used
    :return: Local rate
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "bochenek_20"

    if rate_name not in local_frb_rates.keys():
        raise Exception(f"Rate name '{rate_name}' not recognised. "
                        f"The following rates are available: {local_frb_rates.keys()}")
    else:
        local_rate, ref = local_frb_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    return local_rate.to("Mpc-3 yr-1")

def get_frb_rate(evolution_name=None, rate_name=None, **kwargs):
    """Returns a local rate of core-collapse supernovae (CCSNe) as a function of redshift.

    :param evolution_name: Name of Star Formation evolution to use
    :param rate_name: Name of local FRB rate to be used
    :return: Rate as a function of redshift
    """

    normed_evolution = get_sfr_evolution(evolution_name, **kwargs)
    local_rate = get_local_frb_rate(rate_name)

    return lambda z: normed_evolution(z) * local_rate
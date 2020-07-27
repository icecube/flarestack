import logging
from astropy import units as u

def madau_14(z, **kwargs):
    """Star formation history as a function of redshift, from
    Madau & Dickinson 2014 (http://arxiv.org/abs/1403.0007v3)

    """
    rate = (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
    return rate

def strolger_15(z, **kwargs):
    """
    star formation history
    https://arxiv.org/pdf/1509.06574.pdf

    result is in solar masses/year/Mpc^3

    Can match Figure 6, if the h^3 factor is divided out
    """
    rate = (1+z)**5.0 / (1 + ((1+z)/1.5)**6.1)

    return rate


sfr_evolutions = {
    "madau_14": (madau_14, "http://arxiv.org/abs/1403.0007v3"),
    "strolger_15": (strolger_15, "https://arxiv.org/abs/1509.06574"),
}


def get_sfr_evolution(evolution_name=None, **kwargs):
    """Returns SFR evolution as a function of redshift

    :param evolution_name: Name of chosen evolution
    :return: Normalised evolution, equal to 1 at z=0
    """

    if evolution_name is None:
        logging.info("No evolution specified. Assuming default evolution.")
        evolution_name = "madau_14"

    if evolution_name not in sfr_evolutions.keys():
        raise Exception(f"Evolution name '{evolution_name}' not recognised. "
                        f"The following source evolutions are available: {sfr_evolutions.keys()}")
    else:
        evolution, ref = sfr_evolutions[evolution_name]
        logging.info(f"Loaded evolution '{evolution_name}' ({ref})")

    normed_evolution = lambda x: evolution(x, **kwargs)/evolution(0.0, **kwargs)
    return normed_evolution

local_sfr_rates = {
    "madau_14": (0.015 * u.solMass / (u.Mpc**3 * u.year), "http://arxiv.org/abs/1403.0007v3"),
    "strolger_15": (0.015 * u.solMass/ (u.Mpc**3 * u.year), "https://arxiv.org/abs/1509.06574")
}

def get_local_sfr_rate(rate_name=None):
    """Returns local SFR rate

    :param rate_name: Name of chosen evolution
    :return: Normalised evolution, equal to 1 at z=0
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "madau_14"

    if rate_name not in local_sfr_rates.keys():
        raise Exception(f"Rate name '{rate_name}' not recognised. "
                        f"The following source evolutions are available: {local_sfr_rates.keys()}")
    else:
        local_rate, ref = local_sfr_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    return local_rate.to("solMass Mpc-3 yr-1")

def get_sfr_rate(evolution_name=None, rate_name=None, **kwargs):
    """Load a Star Formation Rate as a function of redshift. This is a product of
    a SFR evolution and a SFR local rate.

    :param evolution_name: Name of SFR evolution to use
    :param rate_name: Name of SFR local rate to use
    :return: TDE Rate function
    """
    normed_evolution = get_sfr_evolution(evolution_name, **kwargs)
    local_rate = get_local_sfr_rate(rate_name)
    return lambda z: local_rate*normed_evolution(z)
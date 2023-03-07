import logging
import numpy as np
from astropy import units as u


def blazar_evolution_ajello_15(z, k=12.14, eta=-0.15, tau=2.79, l_gamma=43):
    """Blazar redshift evolution from Ajello_15 et. al. 2015 (https://arxiv.org/abs/1501.05301).
    As confirmed by Marcotulli et al. 2020 (https://arxiv.org/abs/2006.04703),
    a pure density evolution is now favoured.

    Parameterisation is Equation 5, best-fit parameters in Table 2 (row PDE),
    for a 10^46 erg s^-1 blazar.

    :param z: Redshift
    :param k:
    :param eta:
    :param tau:
    :param l_gamma:
    :return: f(z)
    """

    z = np.array(z)

    kd = k + tau * (l_gamma - 46.)

    res = ((1. + z) ** kd) * np.exp(z / eta)

    return res


blazar_evolutions = {
    "ajello_15": (
        blazar_evolution_ajello_15,
        None,
        None,
        "https://arxiv.org/abs/1501.05301",
    ),
}


def get_blazar_evolution(evolution_name=None, with_range=False, **kwargs):
    """Returns a Blazar evolution as a function of redshift

    :param evolution_name: Name of chosen evolution
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Normalised evolution, equal to 1 at z=0
    """

    if evolution_name is None:
        logging.info("No evolution specified. Assuming default evolution.")
        evolution_name = "ajello_15"

    if evolution_name not in blazar_evolutions.keys():
        raise Exception(
            f"Evolution name '{evolution_name}' not recognised. "
            f"The following source evolutions are available: {blazar_evolutions.keys()}"
        )
    else:
        evolution, lower_ev, upper_ev, ref = blazar_evolutions[evolution_name]
        logging.info(f"Loaded evolution '{evolution_name}' ({ref})")

    def normed_evolution(z):
        """Normalised redshift evolution, defined such that f(z=0.0) = 1..

        :param z: Redshift
        :return: Rate relative to f(z=0.0(
        """
        return evolution(z, **kwargs) / evolution(0.0, **kwargs)

    if with_range:

        if lower_ev is None:
            raise Exception(
                f"No one sigma evolution range found for evolution '{evolution_name}'. "
                f"Use a different rate, or set 'with_range=False'."
            )

        return (
            normed_evolution,
            lambda z: lower_ev(z) / lower_ev(0.0),
            lambda z: upper_ev(z) / upper_ev(0.0),
        )

    else:
        return normed_evolution


def blazar_rate_ajello_15(l_gamma=43, a_raw=1.22, gamma1=2.8, l_star_raw=0.44, gamma2=1.26, mustar=2.22, beta=0.1, sigma=0.28, photon_index=2.2):
    """Blazar redshift evolution from Ajello_15 et. al. 2015 (https://arxiv.org/abs/1501.05301).
    As confirmed by Marcotulli et al. 2020 (https://arxiv.org/abs/2006.04703),
    a pure density evolution is now favoured.

    Parameterisation is Equation 1, best-fit parameters in Table 2 (row PDE),
    for a 10^43 erg s^-1 blazar.

    :param A:
    :param eta:
    :param tau:
    :param l_gamma:
    :return: f(z)
    """

    a = a_raw * 10.**-2 / u.Gpc**3.
    l_star = l_star_raw * 10.**46.

    factor_1 = a / (np.log(10) * 10.**l_gamma)
    factor_2 = (((10.**l_gamma)/l_star)**gamma1 + ((10.**l_gamma)/l_star)**gamma2)**-1.

    mu = mustar + beta * (l_gamma - 46.)

    factor_3 = np.exp(-0.5 * ((photon_index - mu)/sigma)**2.)

    return factor_1 * factor_2 * factor_3


local_blazar_rates = {
    "ajello_15": (
        blazar_rate_ajello_15,
        None,
        None,
        "https://arxiv.org/abs/1706.00391",
    ),
}


def get_local_blazar_rate(rate_name=None, with_range=False, **kwargs):
    """Returns local blazar rate

    :param rate_name: Name of chosen evolution
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Normalised evolution, equal to 1 at z=0
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "ajello_15"

    if rate_name not in local_blazar_rates.keys():
        raise Exception(
            f"Rate name '{rate_name}' not recognised. "
            f"The following source evolutions are available: {local_blazar_rates.keys()}"
        )
    else:
        local_rate, lower_lim, upper_lim, ref = local_blazar_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    if with_range:

        if lower_lim is None:
            raise Exception(
                f"No one sigma rate range found for rate '{rate_name}'. "
                f"Use a different rate, or set 'with_range=False'."
            )

        return (
            local_rate(**kwargs).to("Mpc-3"),
            lower_lim(**kwargs).to("Mpc-3"),
            upper_lim(**kwargs).to("Mpc-3"),
        )

    else:
        return local_rate(**kwargs).to("Mpc-3") / u.s


def get_blazar_rate(evolution_name=None, rate_name=None, with_range=False, **kwargs):
    """Load a blazar rate as a function of redshift. This is a product of
    a blazar evolution and a blazar local rate.

    :param evolution_name: Name of blazar evolution to use
    :param rate_name: Name of blazar local rate to use
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Blazar Rate function
    """
    normed_evolution = get_blazar_evolution(
        evolution_name=evolution_name, with_range=with_range, **kwargs
    )
    local_rate = get_local_blazar_rate(rate_name=rate_name, with_range=with_range, **kwargs)

    if with_range:
        return (
            lambda z: local_rate[0] * normed_evolution[0](z),
            lambda z: local_rate[1] * normed_evolution[1](z),
            lambda z: local_rate[2] * normed_evolution[2](z),
        )
    else:
        return lambda z: local_rate * normed_evolution(z)

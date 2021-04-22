import logging
import numpy as np
from astropy import units as u


def grb_evolution_lien_14(z, z1=3.6, n1=2.07, n2=-0.7):
    """GRB redshift evolution from Lien et. al. 2014 (https://arxiv.org/abs/1311.4567).
    Parameterisation is Equation 1, best-fit parameters in Table 2.

    :param z: Redshift
    :param z1: Break redshift
    :param n1: Low-z index
    :param n2: High-z index
    :return: f(z)
    """

    z = np.array(z)

    mask = z < z1

    res = np.ones_like(z)

    if np.sum(mask) > 0.:

        res[mask] = (1. + z[mask])**n1

    if np.sum(~mask) > 0.:
        res[~mask] = ((1. + z1) ** (n1 - n2)) * (1. + z[~mask])**n2

    return res


def lien_14_lower(z):
    """Lower limit GRB redshift evolution from Lien et. al. 2014 (https://arxiv.org/abs/1311.4567).
    Parameterisation is Equation 1, lower-limit parameters in Table 5.

    :param z: Redshift
    :return: f(z)
    """
    return grb_evolution_lien_14(z, z1=3.6, n1=2.1, n2=-3.5)


def lien_14_upper(z):
    """Upper limit GRB redshift evolution from Lien et. al. 2014 (https://arxiv.org/abs/1311.4567).
    Parameterisation is Equation 1, lower-limit parameters in Table 6.

    :param z: Redshift
    :return: f(z)
    """
    return grb_evolution_lien_14(z, z1=3.6, n1=1.95, n2=0.0)


grb_evolutions = {
    "lien_14": (
        grb_evolution_lien_14,
        lien_14_lower,
        lien_14_upper,
        "https://arxiv.org/abs/1311.4567"
    ),
}


def get_grb_evolution(evolution_name=None, with_range=False, **kwargs):
    """Returns a GRB evolution as a function of redshift

    :param evolution_name: Name of chosen evolution
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Normalised evolution, equal to 1 at z=0
    """

    if evolution_name is None:
        logging.info("No evolution specified. Assuming default evolution.")
        evolution_name = "lien_14"

    if evolution_name not in grb_evolutions.keys():
        raise Exception(f"Evolution name '{evolution_name}' not recognised. "
                        f"The following source evolutions are available: {grb_evolutions.keys()}")
    else:
        evolution, lower_ev, upper_ev, ref = grb_evolutions[evolution_name]
        logging.info(f"Loaded evolution '{evolution_name}' ({ref})")

    def normed_evolution(z):
        """Normalised redshift evolution, defined such that f(z=0.0) = 1..

        :param z: Redshift
        :return: Rate relative to f(z=0.0(
        """
        return evolution(z, **kwargs)/evolution(0.0, **kwargs)

    if with_range:

        if lower_ev is None:
            raise Exception(f"No one sigma evolution range found for evolution '{evolution_name}'. "
                            f"Use a different rate, or set 'with_range=False'.")

        return normed_evolution, lambda z: lower_ev(z)/lower_ev(0.0), lambda z: upper_ev(z)/upper_ev(0.0)

    else:
        return normed_evolution


local_grb_rates = {
    "lien_14": (
        0.42 / (u.Gpc**3 * u.yr),
        0.38 / (u.Gpc**3 * u.yr),
        0.51 / (u.Gpc**3 * u.yr),
        "https://arxiv.org/abs/1706.00391"
    ),
}


def get_local_grb_rate(rate_name=None, with_range=False):
    """Returns local grb rate

    :param rate_name: Name of chosen evolution
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Normalised evolution, equal to 1 at z=0
    """

    if rate_name is None:
        logging.info("No rate specified. Assuming default rate.")
        rate_name = "lien_14"

    if rate_name not in local_grb_rates.keys():
        raise Exception(f"Rate name '{rate_name}' not recognised. "
                        f"The following source evolutions are available: {local_grb_rates.keys()}")
    else:
        local_rate, lower_lim, upper_lim, ref = local_grb_rates[rate_name]
        logging.info(f"Loaded rate '{rate_name}' ({ref})")

    if with_range:

        if lower_lim is None:
            raise Exception(f"No one sigma rate range found for rate '{rate_name}'. "
                            f"Use a different rate, or set 'with_range=False'.")

        return local_rate.to("Mpc-3 yr-1"), lower_lim.to("Mpc-3 yr-1"), upper_lim.to("Mpc-3 yr-1")

    else:
        return local_rate.to("Mpc-3 yr-1")


def get_grb_rate(evolution_name=None, rate_name=None, with_range=False, **kwargs):
    """Load a GRB rate as a function of redshift. This is a product of
    a GRB evolution and a GRB local rate.

    :param evolution_name: Name of GRB evolution to use
    :param rate_name: Name of GRB local rate to use
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: GRB Rate function
    """
    normed_evolution = get_grb_evolution(
        evolution_name=evolution_name,
        with_range=with_range,
        **kwargs
    )
    local_rate = get_local_grb_rate(
        rate_name=rate_name,
        with_range=with_range
    )

    if with_range:
        return lambda z: local_rate[0] * normed_evolution[0](z), \
               lambda z: local_rate[1] * normed_evolution[1](z), \
               lambda z: local_rate[2] * normed_evolution[2](z)
    else:
        return lambda z: local_rate * normed_evolution(z)

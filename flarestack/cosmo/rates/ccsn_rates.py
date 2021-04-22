import logging
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import numpy as np
from flarestack.cosmo.rates.sfr_rates import get_local_sfr_rate, get_sfr_evolution, sfr_evolutions, local_sfr_rates

# Taken from https://arxiv.org/pdf/1509.06574.pdf

sn_subclass_rates = {
    "li_11": ({
        "IIn": 0.064,
        "IIP": 0.52,
        "IIL": 0.073,
        "II": 0.064 + 0.52 + 0.073,
        "Ib": 0.069,
        "Ic": 0.176,
        "Ibc": 0.069 + 0.176,
    }, "https://arxiv.org/abs/1006.4612")
}


def get_sn_subfraction(sn_subclass_fractions_name=None):
    """Return a value of kcc (SN per unit star formation)

    :param sn_subclass_fractions_name: Name of kcc to be used
    :return: Value of kcc
    """

    if sn_subclass_fractions_name is None:
        logging.info("No specified sn_subclass_fractions_name. Assuming default.")
        sn_subclass_fractions_name = "li_11"

    if sn_subclass_fractions_name not in sn_subclass_rates.keys():
        raise Exception(f"Subclass name '{sn_subclass_fractions_name}' not recognised. "
                        f"The following kcc values are available: {sn_subclass_rates.keys()}")
    else:
        sn_rates, ref = sn_subclass_rates[sn_subclass_fractions_name]
        logging.info(f"Loaded SN subclass fractions '{sn_subclass_fractions_name}' ({ref})")

    return sn_rates


def get_sn_fraction(sn_subclass=None, sn_subclass_fractions_name=None):
    """Return SN rates for specific types. These are taken from
    https://arxiv.org/pdf/1509.06574.pdf, and are assumed to be fixed
    fractions of the overall SN rate. Acceptable types are:
        SNIIn
        SNIIP
        SNIb
        SNIc
        SNIbc (equal to Ib + Ic)

    :param sn_subclass: Type of SN
    :param sn_subclass_fractions_name: Name of estimates to use for relative rates of each subclass
    :return: fraction represented by that subtype
    """
    if sn_subclass is None:
        logging.info("No specified subclass of supernova. Assuming overall CCSN rate.")
        return 1.0

    else:

        sn_types = get_sn_subfraction(sn_subclass_fractions_name)

        if sn_subclass in sn_types.keys():
            logging.info(f"Subclass '{sn_subclass}' is equal to "
                         f"{100.*sn_types[sn_subclass]:.2f}% of the CCSN rate.")
            return sn_types[sn_subclass]
        else:
            raise Exception(f"Supernova type '{sn_subclass}' not recognised. "
                            f"The following types are available: {sn_types.keys()}")


kcc_rates = {
    "madau_14": (
        0.0068 / u.solMass,
        None,
        None,
        "http://arxiv.org/abs/1403.0007v3"
    ),
    "strolger_15": (
        0.0091 * cosmo.h ** 2. / cosmo.h ** 3 / u.solMass,
        (0.0091 - 0.0017) * cosmo.h ** 2. / cosmo.h ** 3 / u.solMass,
        (0.0091 + 0.0017) * cosmo.h ** 2. / cosmo.h ** 3 / u.solMass,
        "https://arxiv.org/abs/1509.06574"
    )
}


def get_kcc_rate(kcc_name=None, with_range=False):
    """Return a value of kcc (SN per unit star formation)

    :param kcc_name: Name of kcc to be used
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Value of kcc
    """

    if kcc_name is None:
        logging.info("No specified kcc (sn per unit star formation). Assuming default kcc.")
        kcc_name = "madau_14"

    if kcc_name not in kcc_rates.keys():
        raise Exception(f"kcc name '{kcc_name}' not recognised. "
                        f"The following kcc values are available: {kcc_rates.keys()}")
    else:
        kcc, lower_lim, upper_lim, ref = kcc_rates[kcc_name]
        logging.info(f"Loaded kcc '{kcc_name}' ({ref})")

    if with_range:

        if lower_lim is None:
            logging.warning(f"No one sigma kcc range found for kcc '{kcc_name}'. "
                            f"No error on this value will be propagated. ")

        return kcc, lower_lim, upper_lim

    else:
        return kcc


def get_local_ccsn_rate(rate_name=None, kcc_name=None, sn_subclass=None, with_range=False):
    """Returns a local rate of core-collapse supernovae (CCSNe).

    :param rate_name: Name of local Star Formation Rate (sfr) to be used
    :param kcc_name: Name of kcc (sn per unit star formation) to be used
    :param sn_subclass: Name of subclass of CCSNe to use
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Local rate
    """

    sfr_rate = get_local_sfr_rate(rate_name=rate_name)

    if kcc_name is None:
        if rate_name in kcc_rates:
            kcc_name = rate_name

    kcc = get_kcc_rate(kcc_name=kcc_name, with_range=with_range)

    subclass_fraction = get_sn_fraction(sn_subclass=sn_subclass)

    if with_range:

        return sfr_rate * kcc[0] * subclass_fraction, \
               sfr_rate * kcc[1] * subclass_fraction, \
               sfr_rate * kcc[2] * subclass_fraction
    else:
        return sfr_rate * kcc * subclass_fraction


def get_ccsn_rate(evolution_name=None, rate_name=None, kcc_name=None, sn_subclass=None, with_range=False, **kwargs):
    """Returns a local rate of core-collapse supernovae (CCSNe) as a function of redshift.

    :param evolution_name: Name of Star Formation evolution to use
    :param rate_name: Name of local Star Formation Rate (sfr) to be used
    :param kcc_name: Name of kcc (sn per unit star formation) to be used
    :param sn_subclass: Name of subclass of CCSNe to use
    :param with_range: Boolean to return +/- one sigma range functions alongside central rate
    :return: Rate as a function of redshift
    """

    normed_evolution = get_sfr_evolution(evolution_name=evolution_name, **kwargs)
    local_rate = get_local_ccsn_rate(
        rate_name=rate_name,
        kcc_name=kcc_name,
        sn_subclass=sn_subclass,
        with_range=with_range
    )

    if with_range:
        return lambda z: local_rate[0] * normed_evolution(z), \
               lambda z: local_rate[1] * normed_evolution(z), \
               lambda z: local_rate[2] * normed_evolution(z)
    else:
        return lambda z: local_rate * normed_evolution(z)

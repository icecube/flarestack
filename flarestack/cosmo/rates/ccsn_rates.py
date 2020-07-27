import logging
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from flarestack.cosmo.rates.sfr_rates import get_local_sfr_rate, get_sfr_evolution

# Taken from https://arxiv.org/pdf/1509.06574.pdf

sn_types = {
    "IIn": 0.064,
    "IIP": 0.52,
    "Ib": 0.069,
    "Ic": 0.176,
    "Ibc": 0.069 + 0.176,
    "all": 1.0
}

def get_sn_fraction(sn_subclass=None):
    """Return SN rates for specific types. These are taken from
    https://arxiv.org/pdf/1509.06574.pdf, and are assumed to be fixed
    fractions of the overall SN rate. Acceptable types are:
        SNIIn
        SNIIP
        SNIb
        SNIc
        SNIbc (equal to Ib + Ic)

    :param sn_subclass: Type of SN
    :return: fraction represented by that subtype
    """
    if sn_subclass is None:
        logging.info("No specified subclass of supernova. Assuming overall CCSN rate.")
        sn_subclass = "all"

    if sn_subclass in sn_types.keys():
        logging.info(f"Subclass '{sn_subclass}' is equal to {100.*sn_types[sn_subclass]:.2f}% of the CCSN rate.")
        return sn_types[sn_subclass]
    else:
        raise Exception(f"Supernova type '{sn_subclass}' not recognised. "
                        f"The following types are available: {sn_types.keys()}")

kcc_rates = {
    "madau_14": (0.0068 / u.solMass, "http://arxiv.org/abs/1403.0007v3"),
    "strolger_15": (0.0091 * cosmo.h ** 2. / cosmo.h ** 3 / u.solMass, "https://arxiv.org/abs/1509.06574")
}

def get_kcc_rate(kcc_name=None):
    """Return a value of kcc (SN per unit star formation)

    :param kcc_name: Name of kcc to be used
    :return: Value of kcc
    """

    if kcc_name is None:
        logging.warning("No specified kcc (sn per unit star formation). Assuming default kcc.")
        kcc_name = "madau_14"

    if kcc_name not in kcc_rates.keys():
        raise Exception(f"kcc name '{kcc_name}' not recognised. "
                        f"The following kcc values are available: {kcc_rates.keys()}")
    else:
        kcc, ref = kcc_rates[kcc_name]
        logging.info(f"Loaded kcc '{kcc_name}' ({ref})")

    return kcc


def get_local_ccsn_rate(rate_name=None, kcc_name=None, sn_subclass=None, fraction=1.0):
    """Returns a local rate of core-collapse supernovae (CCSNe).

    :param rate_name: Name of local Star Formation Rate (sfr) to be used
    :param kcc_name: Name of kcc (sn per unit star formation) to be used
    :param sn_subclass: Name of subclass of CCSNe to use
    :param fraction: Fraction of specified rate to include
    :return: Local rate
    """

    sfr_rate = get_local_sfr_rate(rate_name)

    if kcc_name is None:
        if rate_name in kcc_rates:
            kcc_name = rate_name

    kcc = get_kcc_rate(kcc_name)

    subclass_fraction = get_sn_fraction(sn_subclass)

    if fraction != 1.0:
        logging.info(f"Assuming a modified rate that is {100.*fraction} of that total.")


    return sfr_rate * kcc * subclass_fraction * fraction

def get_ccsn_rate(evolution_name=None, rate_name=None, kcc_name=None, sn_subclass=None, fraction=1.0, **kwargs):
    """Returns a local rate of core-collapse supernovae (CCSNe) as a function of redshift.

    :param evolution_name: Name of Star Formation evolution to use
    :param rate_name: Name of local Star Formation Rate (sfr) to be used
    :param kcc_name: Name of kcc (sn per unit star formation) to be used
    :param sn_subclass: Name of subclass of CCSNe to use
    :param fraction: Fraction of specified rate to include
    :return: Rate as a function of redshift
    """
    normed_evolution = get_sfr_evolution(evolution_name, **kwargs)
    local_rate = get_local_ccsn_rate(rate_name, kcc_name, sn_subclass, fraction)

    return lambda z: normed_evolution(z) * local_rate
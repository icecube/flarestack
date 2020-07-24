from astropy.cosmology import Planck15 as cosmo
from flarestack.cosmo.rates.sfr_rates import sfr_madau, sfr_clash_candels

def ccsn_clash_candels(z):
    """Best fit k from paper https://arxiv.org/pdf/1509.06574.pdf"""
    # Why divide by h^3???
    return 0.0091 * sfr_clash_candels(z) * cosmo.h**2. / cosmo.h**3


def ccsn_madau(z):
    """"Best fit k from http://arxiv.org/pdf/1403.0007v3.pdf"""
    return 0.0068 * sfr_madau(z)


def get_sn_fraction(sn_type):
    """Return SN rates for specific types. These are taken from
    https://arxiv.org/pdf/1509.06574.pdf, and are assumed to be fixed
    fractions of the overall SN rate. Acceptable types are:
        SNIIn
        SNIIP
        SNIb
        SNIc
        SNIbc (equal to Ib + Ic)

    :param sn_type: Type of SN
    :return: fraction represented by that subtype
    """
    if sn_type == "IIn":
        return 0.064
    elif sn_type == "IIp":
        return 0.52
    elif sn_type == "Ib":
        return 0.069
    elif sn_type == "Ic":
        return 0.176
    elif sn_type == "Ibc":
        return 0.069 + 0.176
    else:
        raise Exception("SN sn_type " + str(sn_type) + " not recognised!")


def get_sn_type_rate(fraction=1.0, sn_type=None, rate=ccsn_clash_candels):
    """Return SN rates for given fraction of the CCSN rate, or specific types.
    The types are taken from https://arxiv.org/pdf/1509.06574.pdf, and are
    assumed to be fixed fractions of the overall SN rate. Acceptable types are:
        IIn
        IIP
        Ib
        Ic
        Ibc (equal to Ib + Ic)

    :param fraction: Fraction of SN
    :param sn_type: Type of SN
    :param rate: CCSN rate to be used (Clash Candels by default)
    :return: corresponding rate
    """

    if (fraction != 1.0) and (sn_type is not None):
        raise Exception("Type and fraction both specified!")
    elif sn_type is not None:
        return lambda x: get_sn_fraction(sn_type) * rate(x)
    else:
        return lambda x: fraction * rate(x)
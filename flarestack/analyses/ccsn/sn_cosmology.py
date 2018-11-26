from flarestack.utils.neutrino_cosmology import calculate_transient, \
    sfr_madau, sfr_clash_candels
from flarestack.analyses.ccsn.ccsn_limits import limits
from flarestack.core.energy_PDFs import EnergyPDF
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo


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
    if sn_type == "SNIIn":
        return 0.064
    elif sn_type == "SNIIP":
        return 0.52
    elif sn_type == "SN1b":
        return 0.069
    elif sn_type == "SN1c":
        return 0.176
    elif sn_type == "SN1bc":
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

if __name__ == "__main__":

    e_pdf_dict_template = {
        "Name": "Power Law",
        "Source Energy (erg)": 10 ** 49 * u.erg,
        "E Min": 10 ** 2,
        "E Max": 10 ** 7,
        # "Gamma": 2
    }


    results = [
        ["SNIIn", 1.0],
        ["SNIIP", 1.0],
        ["SN1bc", 1.0]
    ]

    for [name, nu_bright] in results:

        def f(x):
            return get_sn_type_rate(sn_type=name)(x) * nu_bright

        e_pdf_dict = dict(e_pdf_dict_template)

        energy_pdf = EnergyPDF.create(e_pdf_dict)

        e_pdf_dict["Source Energy (erg)"] = limits[name]["Fixed Energy (erg)"]

        calculate_transient(e_pdf_dict, f, name, zmax=0.3,
                            nu_bright_fraction=nu_bright,
                            diffuse_fit="Joint")

    # calculate_transient(e_pdf_dict_template, ccsn_clash_candels, "CCSN",
    #                     zmax=2.5, diffuse_fit="Joint")

    # from flarestack.utils.neutrino_cosmology import \
    #     define_cosmology_functions, sfr_clash_candels, sfr_madau
    # from flarestack.analyses.ccsn.ccsn_limits import limit_units
    # import numpy as np
    #
    # rate = get_sn_type_rate(sn_type="SNIIn")
    # rate = ccsn_clash_candels
    # nu_e_flux_1GeV = 3.18575405845223e-09 * limit_units
    #
    # gamma = -2.5
    #
    # rate_per_z, nu_flux_per_z, cumulative_nu_flux = define_cosmology_functions(
    #     rate, nu_e_flux_1GeV, gamma
    # )
    #
    # zref = 0.1
    # print rate(0.0)
    # print rate_per_z(0.3)
    # #
    # calculate_transient(e_pdf_dict_template, ccsn_clash_candels, "CCSN",
    #                     zmax=2.5)
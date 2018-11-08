from flarestack.utils.neutrino_cosmology import calculate_transient, \
    sfr_madau, sfr_clash_candels
from flarestack.core.energy_PDFs import EnergyPDF
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo


def ccsn_clash_candels(z):
    """Best fit k from paper https://arxiv.org/pdf/1509.06574.pdf"""
    return 0.0091 * cosmo.h**2 * sfr_clash_candels(z)


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

    :param sn_type: Fraction of SN
    :param type: Type of SN
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

    # results = [
    #     ["SNIIn", 1.5 * 10 ** 49 * u.erg, 0.064, 1.0],
    #     ["SNIIP", 6 * 10 ** 48 * u.erg, 0.52, 1.0],
    #     ["SN1bc", 4.5 * 10 ** 48 * u.erg, 0.069 + 0.176, 1.0]
    # ]

    e_pdf_dict_template = {
        "Name": "Power Law",
        "Source Energy (erg)": 10 ** 49 * u.erg,
        "E Min": 10 ** 2,
        "E Max": 10 ** 7,
        # "Gamma": diffuse_gamma
    }

    results = [
        ["SNIIn", 1.5 * 10 ** 49 * u.erg, 0.064, 1.0],
        ["SNIIP", 6 * 10 ** 48 * u.erg, 0.52, 1.0],
        ["SN1bc", 4.5 * 10 ** 48 * u.erg, 0.069 + 0.176, 1.0]
    ]

    for [name, nu_e, fraction, nu_bright] in results:

        def f(z):
            return fraction * ccsn_clash_candels(z)


        e_pdf_dict = dict(e_pdf_dict_template)

        energy_pdf = EnergyPDF.create(e_pdf_dict)

        # if diffuse_gamma == 2.5:
        #     nu_e *= 3

        e_pdf_dict["Source Energy (erg)"] = nu_e * 4

        calculate_transient(e_pdf_dict, f, name, zmax=0.3,
                            nu_bright_fraction=nu_bright)

    calculate_transient(e_pdf_dict_template, ccsn_clash_candels, "CCSN",
                        zmax=2.5)
from flarestack.utils.neutrino_cosmology import calculate_transient
from astropy import units as u

e_pdf_dict_template = {
    "Name": "Power Law",
    "Source Energy (erg)": 1.0 * 10**54 * u.erg,
    "E Min": 10**2,
    "E Max": 10**7,
    "Gamma": 2.0
}

# Assumed source evolution is highly negative
non_jetted_m = 0

eta = -2


def raw_tde_evolution(z):
    """https://arxiv.org/abs/1509.01592"""
    evolution = ((1 + z)**(0.2 * eta) + ((1 + z)/1.43)**(-3.2 * eta) +
                 ((1 + z)/2.66)**(-7 * eta)
                 )**(1./eta)
    return evolution


def normed_tde_evolution(z):
    """TDE evolution, scaled such that normed_evolution(0) = 1. This function
    can then be multiplied by the local TDE density to give overall TDE
    redshift evolution

    :param z: Redshift
    :return: Normalised scaling of density with redshift
    """
    return raw_tde_evolution(z)/raw_tde_evolution(0.0)


def tde_rate(z):
    """TDE rate as a function of redshift, equal to the local density
    multiplied by the redshift evolution of the density

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return normed_tde_evolution(z) * 10**-7 / (u.Mpc**3 * u.yr)


res = [
    ("gold", 8 * 10**50 * u.erg),
    ("obscured", 4.5*10**51 * u.erg),
    ("silver", 3 * 10**49 * u.erg)
]

for (name, energy) in res:

    e_pdf_dict = dict(e_pdf_dict_template)
    e_pdf_dict["Source Energy (erg)"] = energy

    calculate_transient(e_pdf_dict, tde_rate, name + " TDEs",
                        zmax=6.0, diffuse_fit="Northern Tracks")

# Assumed source evolution is highly negative
jetted_m = -3


def biehl_jetted_rate(z):
    """Rate of TDEs assumed by Biehl et al. 2018 is 0.1 per Gpc per year
    (10^-10 per Mpc per year). The source evolution is assumed to be
    negative, with an index m=3, though the paper also considers indexes up
    to m=0 (flat). More details found under https://arxiv.org/abs/1711.03555

    :param z: Redshift
    :return: Jetted TDE rate
    """
    rate = 0.1 * (1 + z)**jetted_m / (u.Gpc**3 * u.yr)
    return rate.to("Mpc-3 yr-1")


def standard_jetted_rate(z):
    return normed_tde_evolution(z) * 3 * 10**-11 / (u.Mpc**3 * u.yr)


calculate_transient(e_pdf_dict_template, standard_jetted_rate, "jetted TDEs",
                    zmax=2.5)

calculate_transient(e_pdf_dict_template, biehl_jetted_rate,
                    "jetted TDEs (biehl)", zmax=2.5,
                    diffuse_fit="Northern Tracks"
                    )



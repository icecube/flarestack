from __future__ import division
from flarestack.utils.neutrino_cosmology import calculate_transient_cosmology
from astropy import units as u
import matplotlib.pyplot as plt
from flarestack.shared import plot_output_dir
import numpy as np
import os
from flarestack.misc.convert_diffuse_flux_contour import contour_95, \
    upper_contour, lower_contour, global_fit_e_range
from flarestack.utils.neutrino_cosmology import get_diffuse_flux_at_1GeV
from flarestack.analyses.tde.shared_TDE import tde_cat_limit

e_pdf_dict_template = {
    "Name": "Power Law",
    # "Source Energy (erg)": 1.0 * 10**54 * u.erg,
    "E Min": 10**2,
    "E Max": 10**7,
    # "Gamma": 2.0
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
    return raw_tde_evolution(z) / raw_tde_evolution(0.0)



def tde_rate(z):
    """TDE rate as a function of redshift, equal to the local density
    multiplied by the redshift evolution of the density. Rate is derived
    from https://arxiv.org/pdf/1707.03458

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return (normed_tde_evolution(z) * 8 * 10**-7) / (u.Mpc**3 * u.yr)


def theoretical_tde_rate(z):
    """Optimistic TDE rate from https://arxiv.org/pdf/1601.06787 as a function
    of redshift, equal to the local density multiplied by the redshift
    evolution of the density

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return (normed_tde_evolution(z) * 1.5 * 10**-6) / (u.Mpc**3 * u.yr)


res = [
    ("gold", 8 * 10**50 * u.erg, "Non-jetted TDEs"),
    # ("obscured", 4.5*10**51 * u.erg),
    # ("silver", 3 * 10**49 * u.erg)
]

norms = dict()

for (name, energy, key) in res:

    class_dict = dict()

    # for i, rate in enumerate([tde_rate, theoretical_tde_rate]):
    for i, rate in enumerate([tde_rate]):
        e_pdf_dict = dict(e_pdf_dict_template)
        e_pdf_dict["Source Energy (erg)"] = tde_cat_limit(name, 2.5) * u.erg

        rate_key = ["(Observed Rate)", "(Theoretical Rate)"][i]

        class_dict[rate_key] = calculate_transient_cosmology(e_pdf_dict, rate,
                                                             name + "TDEs",
                                                             zmax=6.0, diffuse_fit="Joint")

    norms[key] = class_dict

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
    """Rate taken from appendix of https://arxiv.org/pdf/1706.00391"""
    return (normed_tde_evolution(z) * 3 * 10**-11) / (u.Mpc**3 * u.yr)

#
norms["Jetted TDEs"] = dict()

e_pdf_dict = dict(e_pdf_dict_template)
e_pdf_dict["Source Energy (erg)"] = tde_cat_limit("jetted", 2.5) * u.erg
# for i, rate in enumerate([standard_jetted_rate, biehl_jetted_rate]):
for i, rate in enumerate([standard_jetted_rate]):
    rate_key = ["(Observed Rate)", "(Theoretical Rate)"][i]
    norms["Jetted TDEs"][rate_key] = calculate_transient_cosmology(e_pdf_dict,
                                                                   rate,
                                               "jetted TDEs", zmax=2.5,
                                                                   diffuse_fit="Joint")


base_dir = plot_output_dir("analyses/tde/")

e_range = np.logspace(2.73, 5.64, 3)

try:
    os.makedirs(base_dir)
except OSError:
    pass


def z(energy, norm):
    return norm * energy ** -0.5


plt.figure()

# Plot 95% contour

plt.fill_between(
    global_fit_e_range,
    global_fit_e_range ** 2 * upper_contour(global_fit_e_range, contour_95),
    global_fit_e_range ** 2 * lower_contour(global_fit_e_range, contour_95),
    color="k", label='IceCube diffuse flux\nApJ 809, 2015',
    alpha=.5,
)

diffuse_norm, diffuse_gamma = get_diffuse_flux_at_1GeV()


plt.plot(global_fit_e_range,
         diffuse_norm * global_fit_e_range**(2. - diffuse_gamma),
         color="k")

linestyles = {
    "(Observed Rate)": "-",
    "(Theoretical Rate)": ":"
}


for i, (name, res_dict) in enumerate(norms.items()):
    # plt.plot(e_range, z(e_range, norm), label=name)
    for (rate, norm) in res_dict.items():
        plt.errorbar(e_range, z(e_range, norm).value,
                     yerr=.25 * np.array([x.value for x in z(e_range, norm)]),
                     uplims=True, label=name,
                     linestyle = linestyles[rate],
                     color=["blue", "orange"][i])

plt.yscale("log")
plt.xscale("log")
plt.legend(loc="lower left")
# plt.title(r"Contribution of TDEs to the Diffuse Neutrino Flux")
plt.ylabel(r"$E^{2}\frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
plt.xlabel(r"$E_{\nu}$ [GeV]")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(base_dir + "diffuse_flux_global_fit.pdf")
plt.close()

# calculate_transient(e_pdf_dict_template, biehl_jetted_rate,
#                     "jetted TDEs (biehl)", zmax=2.5,
#                     # diffuse_fit="Northern Tracks"
#                     )



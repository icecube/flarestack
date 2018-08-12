import astropy
from astropy import units as u
import numpy as np
import math
from shared import catalogue_dir
from astropy.coordinates import Distance

e_0 = 1 * u.GeV


def fluence_integral(gamma, e_min=100*u.GeV, e_max=10*u.PeV):
    e_min = e_min.to(u.GeV)
    e_max = e_max.to(u.GeV)
    if gamma == 2:
        e_integral = np.log10(e_max / e_min) * (u.GeV ** 2)
    else:
        power = 2 - gamma

        # Get around astropy power rounding error (does not give
        # EXACTLY 2)

        e_integral = ((1. / power) * (e_0 ** gamma) * (
                (e_max ** power) - (e_min ** power))
                      ).value * u.GeV ** 2

    return e_integral


def calculate_astronomy(flux, e_pdf_dict, catalogue):

    gamma = e_pdf_dict["Gamma"]

    astro_res = dict()

    # Set the minimum and maximum energy for integration/detection

    if "E Min" in e_pdf_dict.keys():
        e_min = e_pdf_dict["E Min"] * u.GeV
    else:
        e_min = (100 * u.GeV)

    if "E Max" in e_pdf_dict.keys():
        e_max = e_pdf_dict["E Max"] * u.GeV
    else:
        e_max = (10 * u.PeV).to(u.GeV)

    # Set parameters for conversion from CR luminosity to nu luminosity
    f_pi = 0.1
    waxmann_bachall = (3./8.) * f_pi

    f_cr_to_nu = 0.05

    # Integrate over flux to get dN/dt

    phi_power = 1 - gamma

    phi_integral = ((1. / phi_power) * (e_0 ** gamma) * (
            (e_max ** phi_power) - (e_min ** phi_power))).value * u.GeV

    # Integrate over energy to get dE/dt

    if gamma == 2:
        e_integral = np.log(e_max / e_min) * (u.GeV ** 2)
    else:
        power = 2 - gamma

        # Get around astropy power rounding error (does not give
        # EXACTLY 2)

        e_integral = ((1. / power) * (
                (e_max ** power) - (e_min ** power))
                      ).value * u.GeV ** 2

    # Calculate fluence

    tot_fluence = (flux * e_integral) / (u. GeV * u.cm ** 2 * u.s)

    astro_res["Total Fluence (GeV cm^{-2} s^{-1})"] = tot_fluence.value

    print "Total Fluence", tot_fluence

    src_1 = catalogue[0]

    dist_weight = src_1["Distance (Mpc)"]**-2 / np.sum(
        catalogue["Distance (Mpc)"]**-2)

    si = flux * dist_weight / (u. GeV * u.cm ** 2 * u.s)

    astro_res["Flux (per source)"] = si

    lumdist = src_1["Distance (Mpc)"] * u.Mpc

    area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)

    dNdA = (si * phi_integral).to(u.s ** -1 * u.cm ** -2)

    # int_dNdA += dNdA

    N = dNdA * area

    print "There would be", '{:.3g}'.format(N), "neutrinos emitted."
    print "The energy range was assumed to be between", e_min,
    print "and", e_max, "with a spectral index of", gamma

    etot = (si * area * e_integral).to(u.erg/u.s)

    astro_res["Mean Luminosity (erg/s)"] = etot.value

    print "The required neutrino luminosity was", etot

    cr_e = etot / f_cr_to_nu

    print "Assuming", '{:.3g}'.format(100 * f_cr_to_nu),
    print "% was transferred from CR to neutrinos,",
    print "we would require a total CR luminosity of", cr_e

    return astro_res
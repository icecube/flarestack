from __future__ import print_function
from __future__ import division
import astropy
from astropy import units as u
import numpy as np
import math
from flarestack.shared import catalogue_dir
from astropy.coordinates import Distance
from flarestack.core.injector import Injector
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.utils.catalogue_loader import calculate_source_weight

e_0 = 1 * u.GeV

# Set parameters for conversion from CR luminosity to nu luminosity
f_pi = 0.1
waxmann_bachall = (3. / 8.) * f_pi

f_cr_to_nu = 0.05


def find_zfactor(distance):
    """For a given distance, converts this distance to a redshift, and then
    returns the 1+z factor for that distance

    :param distance: Astropy distance (with units)
    :return: Corresponding 1+z factor
    """
    redshift = astropy.coordinates.Distance(distance).compute_z()
    zfactor = 1 + redshift
    return zfactor


# def fluence_integral(gamma, e_min=100*u.GeV, e_max=10*u.PeV):
#     """Performs an integral for fluence over a given energy range. This is
#     the integral of E*
#
#     :param gamma:
#     :param e_min:
#     :param e_max:
#     :return:
#     """
#     e_min = e_min.to(u.GeV)
#     e_max = e_max.to(u.GeV)
#     if gamma == 2:
#         e_integral = np.log(e_max / e_min) * (u.GeV ** 2)
#     else:
#         power = 2 - gamma
#
#         # Get around astropy power rounding error (does not give
#         # EXACTLY 2)
#
#         e_integral = ((1. / power) * (e_0 ** gamma) * (
#                 (e_max ** power) - (e_min ** power))
#                       ).value * u.GeV ** 2
#
#     return e_integral


def calculate_astronomy(flux, e_pdf_dict, catalogue):

    flux /= (u. GeV * u.cm ** 2 * u.s)

    energy_PDF = EnergyPDF.create(e_pdf_dict)

    astro_res = dict()

    phi_integral = energy_PDF.flux_integral() * u.GeV

    e_integral = energy_PDF.fluence_integral() * u.GeV**2

    # Calculate fluence

    tot_fluence = (flux * e_integral)

    astro_res["Total Fluence (GeV cm^{-2} s^{-1})"] = tot_fluence.value

    print("Total Fluence", tot_fluence)

    src_1 = np.sort(catalogue, order="distance_mpc")[0]

    frac = calculate_source_weight(src_1)/calculate_source_weight(catalogue)

    si = flux * frac

    astro_res["Flux from nearest source"] = si

    print("Total flux:", flux)
    print("Fraction from nearest source:", frac)
    print("Flux from nearest source:", flux * frac)

    lumdist = src_1["distance_mpc"] * u.Mpc

    area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)

    dNdA = (si * phi_integral).to(u.s ** -1 * u.cm ** -2)

    # int_dNdA += dNdA

    N = dNdA * area

    print("There would be", '{:.3g}'.format(N), "neutrinos emitted.")
    print("The energy range was assumed to be between {0} and {1}".format(
        energy_PDF.integral_e_min, energy_PDF.integral_e_max
    ))
    # Energy requires a 1/(1+z) factor

    zfactor = find_zfactor(lumdist)
    etot = (si * area * e_integral).to(u.erg /u.s) * zfactor

    astro_res["Mean Luminosity (erg/s)"] = etot.value

    print("The required neutrino luminosity was", etot)

    cr_e = etot / f_cr_to_nu

    print("Assuming", '{:.3g}'.format(100 * f_cr_to_nu), end=' ')
    print("% was transferred from CR to neutrinos,", end=' ')
    print("we would require a total CR luminosity of", cr_e)

    return astro_res


# def calculate_neutrinos(source, season, inj_kwargs):
#
#     print source
#     inj = Injector(season, [source], **inj_kwargs)
#
#     print "\n"
#     print source["Name"], season["Name"]
#     print "\n"
#
#     energy_pdf = inj_kwargs["Injection Energy PDF"]
#
#     energy = energy_pdf["Energy Flux"] * inj.time_pdf.effective_injection_time(
#         source)
#     print "Neutrino Energy is", energy
#
#     lumdist = source["Distance (Mpc)"] * u.Mpc
#
#     print "Source is", lumdist, "away"
#
#     area = 4 * math.pi * (lumdist.to(u.cm)) ** 2
#
#     nu_flu = energy.to(u.GeV)/area
#
#     print "Neutrino Fluence is", nu_flu
#
#     # if "E Min" in energy_pdf.keys():
#     #     e_min = energy_pdf["E Min"] * u.GeV
#     # else:
#     #     e_min = (100 * u.GeV)
#     #
#     # if "E Max" in energy_pdf.keys():
#     #     e_max = energy_pdf["E Max"] * u.GeV
#     # else:
#     #     e_max = (10 * u.PeV).to(u.GeV)
#
#     e_int = energy_pdf.fluence_integral()
#
#     flux_1GeV = nu_flu/e_int
#
#     print "Flux at 1GeV would be", flux_1GeV, "\n"
#
#     source_mc, omega, band_mask = inj.select_mc_band(inj._mc, source)
#
#     source_mc["ow"] = flux_1GeV * (inj.mc_weights[band_mask] / omega)
#     n_inj = np.sum(source_mc["ow"])
#
#     print ""
#
#     print "This corresponds to", n_inj, "neutrinos"
#
#     return n_inj


def calculate_neutrinos(source, season, inj_kwargs):

    print(source)
    inj = Injector(season, [source], **inj_kwargs)
    energy_pdf = inj_kwargs["Injection Energy PDF"]

    print("\n")
    print(source["Name"], season["Name"])
    print("\n")

    if "Flux at 1GeV" not in list(energy_pdf.keys()):
        # if "Source Energy (erg)" in energy_pdf.keys():
        energy_pdf["Flux at 1GeV"] = \
            energy_pdf["Source Energy (erg)"] / energy_pdf.fluence_integral()

    flux_1_gev = energy_pdf["Flux at 1GeV"] * \
                inj.time_pdf.effective_injection_time(source) * u.s

    print(energy_pdf["Flux at 1GeV"])

    print("Flux at 1GeV would be", flux_1_gev, "\n")
    print(
        "Time is {0} years".format(
            inj.time_pdf.effective_injection_time(source) / (60*60*24*365))
    )
    print("Raw Flux is", energy_pdf["Flux at 1GeV"])

    source_mc, omega, band_mask = inj.select_mc_band(inj._mc, source)

    # northern_mask = inj._mc["sinDec"] > 0.0

    # print "OneWights:", np.sum(inj._mc["ow"])
    # print np.sum(inj._mc["ow"] * inj._mc["trueE"]**-energy_pdf["Gamma"])
    # print np.sum(inj.mc_weights * flux_1_gev)
    #
    # print "Now Ludwig-style:",
    #
    # print np.sum(flux_1_gev * inj.mc_weights[northern_mask])

    source_mc["ow"] = flux_1_gev * inj.mc_weights[band_mask] / omega
    n_inj = np.sum(source_mc["ow"])

    print("")

    print("This corresponds to", n_inj, "neutrinos")

    return n_inj

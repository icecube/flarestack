import logging
import astropy
from astropy import units as u
import numpy as np
import math
from astropy.coordinates import Distance
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.utils.catalogue_loader import (
    get_relative_source_weight,
)

logger = logging.getLogger(__name__)

e_0 = 1 * u.GeV

# Set parameters for conversion from CR luminosity to nu luminosity
f_pi = 0.1
waxmann_bachall = (3.0 / 8.0) * f_pi

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


def calculate_source_astronomy(
    total_flux, phi_integral, e_integral, f_cr_to_nu, catalogue, source
) -> dict:
    astro_dict = dict()
    frac = get_relative_source_weight(catalogue, source)
    si = total_flux * frac

    lumdist = source["distance_mpc"] * u.Mpc
    area = 4 * math.pi * (lumdist.to(u.cm)) ** 2
    dNdA = (si * phi_integral).to(u.s**-1 * u.cm**-2)
    N = dNdA * area

    # Energy requires a 1/(1+z) factor
    zfactor = find_zfactor(lumdist)
    etot = (si * area * e_integral).to(u.erg / u.s) * zfactor

    cr_e = etot / f_cr_to_nu

    astro_dict["frac"] = frac
    astro_dict["flux"] = si
    astro_dict["n_nu"] = N
    astro_dict["E_tot"] = etot
    astro_dict["cr_e"] = cr_e

    return astro_dict


def calculate_astronomy(flux, e_pdf_dict, catalogue) -> dict():
    logger.debug(f"Calculating astronomy for total flux: {flux}")

    # result
    astro_res = dict()

    # convert total flux to number
    # what's the result type / units?
    total_flux = flux / (u.GeV * u.cm**2 * u.s)
    print(total_flux)

    # get flux and energy integrals
    energy_PDF = EnergyPDF.create(e_pdf_dict)

    phi_integral = energy_PDF.flux_integral() * u.GeV

    e_integral = energy_PDF.fluence_integral() * u.GeV**2

    # calculate fluence
    # is this correct to call this a fluence?
    total_fluence = total_flux * e_integral
    print(total_fluence)

    # building the result
    logger.debug("Energy Flux:{0}".format(total_fluence))
    astro_res["Energy flux (GeV cm^{-2} s^{-1})"] = total_fluence.value

    # getting nearest source
    src_1 = np.sort(catalogue, order="distance_mpc")[0]

    src_astro = calculate_source_astronomy(
        total_flux, phi_integral, e_integral, f_cr_to_nu, catalogue, source=src_1
    )

    logger.debug(f"Fraction of total flux from nearest source: {src_astro['frac']}")
    logger.debug(f"Flux from nearest source: {src_astro['flux']}")
    logger.debug(f"There would be {src_astro['n_nu']:.3g} neutrinos emitted.")
    logger.debug(
        f"The energy range was assumed to be between {energy_PDF.integral_e_min} and {energy_PDF.integral_e_max}."
    )
    logger.debug(f"The required neutrino luminosity was {src_astro['E_tot']}.")

    astro_res["Flux from nearest source"] = src_astro["flux"].value
    astro_res["Mean Luminosity (erg/s)"] = src_astro["E_tot"].value
    astro_res["CR luminosity"] = src_astro["cr_e"].value

    logger.debug(
        f"Assuming {100 * f_cr_to_nu:.3g}% was transferred from CR to neutrinos, we would require a total CR luminosity of {src_astro['cr_e']}"
    )

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
#     energy = energy_pdf["Energy Flux"] * inj.sig_time_pdf.effective_injection_time(
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


# def calculate_neutrinos(source, season, inj_kwargs):
#
#     inj = season.make_injector([source], **inj_kwargs)
#     energy_pdf = inj_kwargs["Injection Energy PDF"]
#
#     print("\n")
#     print(source["Name"], season["Name"])
#     print("\n")
#
#     if "Flux at 1GeV" not in list(energy_pdf.keys()):
#         # if "Source Energy (erg)" in energy_pdf.keys():
#         energy_pdf["Flux at 1GeV"] = \
#             energy_pdf["Source Energy (erg)"] / energy_pdf.fluence_integral()
#
#     flux_1_gev = energy_pdf["Flux at 1GeV"] * \
#                 inj.time_pdf.effective_injection_time(source) * u.s
#
#     print(energy_pdf["Flux at 1GeV"])
#
#     print("Flux at 1GeV would be", flux_1_gev, "\n")
#     print(
#         "Time is {0} years".format(
#             inj.time_pdf.effective_injection_time(source) / (60*60*24*365))
#     )
#     print("Raw Flux is", energy_pdf["Flux at 1GeV"])
#
#     source_mc, omega, band_mask = inj.select_mc_band(inj._mc, source)
#
#     # northern_mask = inj._mc["sinDec"] > 0.0
#
#     # print "OneWights:", np.sum(inj._mc["ow"])
#     # print np.sum(inj._mc["ow"] * inj._mc["trueE"]**-energy_pdf["Gamma"])
#     # print np.sum(inj.mc_weights * flux_1_gev)
#     #
#     # print "Now Ludwig-style:",
#     #
#     # print np.sum(flux_1_gev * inj.mc_weights[northern_mask])
#
#     source_mc["ow"] = flux_1_gev * inj.mc_weights[band_mask] / omega
#     n_inj = np.sum(source_mc["ow"])
#
#     print("")
#
#     print("This corresponds to", n_inj, "neutrinos")
#
#     return n_inj

import logging
import astropy
from astropy import units as u
import numpy as np
import math
from astropy.coordinates import Distance
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.utils.catalogue_loader import calculate_source_weight

logger = logging.getLogger(__name__)

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


def calculate_astronomy(flux, e_pdf_dict, catalogue):

    flux /= (u. GeV * u.cm ** 2 * u.s)

    energy_PDF = EnergyPDF.create(e_pdf_dict)

    astro_res = dict()

    phi_integral = energy_PDF.flux_integral() * u.GeV

    e_integral = energy_PDF.fluence_integral() * u.GeV**2

    # Calculate fluence

    tot_fluence = (flux * e_integral)

    astro_res["Energy Flux (GeV cm^{-2} s^{-1})"] = tot_fluence.value

    logger.debug("Energy Flux:{0}".format(tot_fluence))

    src_1 = np.sort(catalogue, order="distance_mpc")[0]

    frac = calculate_source_weight(src_1)/calculate_source_weight(catalogue)

    si = flux * frac

    astro_res["Flux from nearest source"] = si

    logger.debug("Total flux: {0}".format(flux))
    logger.debug("Fraction from nearest source: {0}".format(frac))
    logger.debug("Flux from nearest source: {0}".format(flux * frac))

    lumdist = src_1["distance_mpc"] * u.Mpc

    area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)

    dNdA = (si * phi_integral).to(u.s ** -1 * u.cm ** -2)

    # int_dNdA += dNdA

    N = dNdA * area

    logger.debug("There would be {:.3g} neutrinos emitted.".format(N))
    logger.debug("The energy range was assumed to be between {0} and {1}".format(
        energy_PDF.integral_e_min, energy_PDF.integral_e_max
    ))
    # Energy requires a 1/(1+z) factor

    zfactor = find_zfactor(lumdist)
    etot = (si * area * e_integral).to(u.erg /u.s) * zfactor

    astro_res["Mean Luminosity (erg/s)"] = etot.value

    logger.debug("The required neutrino luminosity was {0}".format(etot))

    cr_e = etot / f_cr_to_nu

    logger.debug(
        "Assuming {0:.3g}% was transferred from CR to neutrinos, we would require a total CR luminosity of {1}".format(
        100 * f_cr_to_nu, cr_e
        )
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

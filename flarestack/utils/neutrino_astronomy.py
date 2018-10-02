import astropy
from astropy import units as u
import numpy as np
import math
from flarestack.shared import catalogue_dir
from astropy.coordinates import Distance
from flarestack.core.injector import Injector

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


def fluence_integral(gamma, e_min=100*u.GeV, e_max=10*u.PeV):
    """Performs an integral for fluence over a given energy range. This is
    the integral of E*

    :param gamma:
    :param e_min:
    :param e_max:
    :return:
    """
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

    # Energy requires a 1/(1+z) factor

    zfactor = find_zfactor(lumdist)
    etot = (si * area * e_integral).to(u.erg/u.s) * zfactor

    astro_res["Mean Luminosity (erg/s)"] = etot.value

    print "The required neutrino luminosity was", etot

    cr_e = etot / f_cr_to_nu

    print "Assuming", '{:.3g}'.format(100 * f_cr_to_nu),
    print "% was transferred from CR to neutrinos,",
    print "we would require a total CR luminosity of", cr_e

    return astro_res


def calculate_neutrinos(source, season, inj_kwargs):
    inj = Injector(season, [source], **inj_kwargs)

    print "\n"
    print source["Name"], season["Name"]
    print "\n"

    lumdist = source["Distance (Mpc)"] * u.Mpc

    print "Source is", lumdist, "away"

    energy_pdf = inj_kwargs["Injection Energy PDF"]

    energy = energy_pdf["Energy Flux"] * inj.time_pdf.effective_injection_time(
        source)
    print "Neutrino Energy is", energy

    area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)

    nu_flu = energy.to(u.GeV)/area

    print "Neutrino Fluence is", nu_flu

    if "E Min" in energy_pdf.keys():
        e_min = energy_pdf["E Min"] * u.GeV
    else:
        e_min = (100 * u.GeV)

    if "E Max" in energy_pdf.keys():
        e_max = energy_pdf["E Max"] * u.GeV
    else:
        e_max = (10 * u.PeV).to(u.GeV)

    e_int = fluence_integral(energy_pdf["Gamma"], e_min, e_max)

    flux_1GeV = nu_flu/e_int

    print "Flux at 1GeV would be", flux_1GeV, "\n"

    source_mc, omega, band_mask = inj.select_mc_band(inj._mc, source)

    source_mc["ow"] = flux_1GeV * (inj.mc_weights[band_mask] / omega)
    n_inj = np.sum(source_mc["ow"])

    print ""

    print "This corresponds to", n_inj, "neutrinos"

    return n_inj

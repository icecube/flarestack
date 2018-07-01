import astropy
from astropy import units as u
import numpy as np
import math
from shared import catalogue_dir
from astropy.coordinates import Distance

# sens_int = 0.089
# # sens_int = 2.5
# cat_path = catalogue_dir + "TDEs/TDE_jetted_catalogue.npy"
# # cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
# # gamma = 2.3
# sens_int = 0.07
# gamma = 2.0

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

    # Sets parameter for conversion of CR to X-Ray luminosity
    baryon_loading = 100

    phi_power = 1 - gamma

    phi_integral = ((1. / phi_power) * (e_0 ** gamma) * (
            (e_max ** phi_power) - (e_min ** phi_power))).value * u.GeV

    if gamma == 2:
        e_integral = np.log(e_max / e_min) * (u.GeV ** 2)
    else:
        power = 2 - gamma

        # Get around astropy power rounding error (does not give
        # EXACTLY 2)

        # e_integral = ((1. / power) * (e_0 ** gamma) * (
        #         (e_max ** power) - (e_min ** power))
        #               ).value * u.GeV ** 2

        e_integral = ((1. / power) * (
                (e_max ** power) - (e_min ** power))
                      ).value * u.GeV ** 2

        # print e_integral, raw_input()

    print "In total", (flux * phi_integral / (
            u.GeV * u.cm ** 2 * u.s)).to(u.s ** -1 * u.cm ** -2)

    int_dNdA = 0.

    tot_fluence = (flux * e_integral) / (u. GeV * u.cm ** 2 * u.s)

    astro_res["Total Fluence (GeV^{-1} cm^{-2} s^{-1})"] = tot_fluence.value

    print "Total Fluence", tot_fluence

    src_1 = catalogue[0]

    dist_weight = src_1["Distance (Mpc)"]**-2 / np.sum(
        catalogue["Distance (Mpc)"]**-2)

    si = flux * dist_weight / (u. GeV * u.cm ** 2 * u.s)

    astro_res["Flux (per source)"] = si

    lumdist = src_1["Distance (Mpc)"] * u.Mpc

    # lumdist = Distance(z=0.3543)

    area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)

    dNdA = (si * phi_integral).to(u.s ** -1 * u.cm ** -2)

    int_dNdA += dNdA

    N = dNdA * area

    print "There would be", '{:.3g}'.format(N), "neutrinos emitted."
    print "The energy range was assumed to be between", e_min,
    print "and", e_max, "with a spectral index of", gamma

    etot = (si * area * e_integral).to(u.erg/u.s)

    astro_res["Total Luminosity (erg/s)"] = etot.value

    print "The required neutrino luminosity was", etot

    cr_e = etot / f_cr_to_nu

    print "Assuming", '{:.3g}'.format(100 * f_cr_to_nu),
    print "% was transferred from CR to neutrinos,",
    print "we would require a total CR luminosity of", cr_e

    print "With a baryonic loading factor of", baryon_loading,
    print "we would expect an X-Ray luminosity of ", cr_e/baryon_loading

    return astro_res

    # for i, source in enumerate(catalogue):
    #     print
    #
    #     print source["Name"],
    #
    #     # si = flux_int * source_weights[i]/(
    #     #         np.sum(source_weights) * u. GeV * u.cm ** 2)
    #
    #
    #     dist_weight = source["Distance (Mpc)"]**-2 / np.sum(
    #         catalogue["Distance (Mpc)"]**-2)
    #
    #     print dist_weight
    # #
    #     si = flux_int * source["Relative Injection Weight"] * dist_weight/(
    #             u.GeV * u.cm ** 2)
    #
    #     lumdist = source["Distance (Mpc)"] * u.Mpc
    #
    #     # lumdist = Distance(z=0.3543)
    #
    #     print lumdist, "\n"
    #
    #     area = (4 * math.pi * (lumdist.to(u.cm)) ** 2)
    #
    #     dNdA = (si * phi_integral).to(u.cm ** -2)
    #
    #     int_dNdA += dNdA
    #
    #     N = dNdA * area
    #
    #     print "dNdA", dNdA, "sum", int_dNdA
    #
    #     print "In total, there were", '{:.3g}'.format(N), "neutrinos emitted."
    #     print "The energy range was assumed to be between", e_min,
    #     print "and", e_max, "with a spectral index of", gamma
    #
    #     etot = (si * area * e_integral).to(u.erg)
    #
    #     print "The required neutrino energy was", etot
    #
    #     cr_e = etot / f_cr_to_nu
    #
    #     print "Assuming", '{:.3g}'.format(100 * f_cr_to_nu),
    #     print "% was transferred from CR to neutrinos,",
    #     print "we would require a total CR energy of", cr_e
    #
    #     print "With a baryonic loading factor of", baryon_loading,
    #     print "we would expect an X-Ray energy of ", cr_e/baryon_loading

    return tot_fluence


# e_dict = {
#     "E Min": 10000
# }

# fluence = 10**-11 * (u.TeV.to(u.GeV) * u.cm**-2)
# integral = fluence_integral(1.8, 100*u.GeV, 10*u.PeV)
#
# print fluence, fluence/integral


# weights_20 = np.array([3.62, 0.34, 0.04])
# weights_23 = np.array([0.887177482715, 0.058582810408, 0.00301094576901])
# calculate_astronomy(sens_int, gamma, cat_path)
# calculate_astronomy(sens_int, 2.3, cat_path, weights_23)
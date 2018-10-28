from astropy import units as u
import astropy
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
import numpy as np
import os
from flarestack.shared import plots_dir
from flarestack.utils.neutrino_astronomy import fluence_integral, \
    calculate_neutrinos_from_flux
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.data.icecube.gfu.gfu_v002_p01 import gfu_dict

# IceCube Diffuse Flux best Fit @ 100TeV

# diffuse_flux = 0.90 * 10**-18 * u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1
#
# # IceCube Joint Best Fit
#
diffuse_flux = 6.7 * 10**-18 * u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1

# diffuse_flux = 2.11872603e-05 * u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1

# code_ref_flux = 1. * u.GeV
# diffuse_ref_flux = (100. * u.TeV)

# IceCube Joint Best Fit

diffuse_flux = 1.01 * 10**-18 * u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1



# Best Fit Spectral Index

# diffuse_gamma = 2.13
#
diffuse_gamma = 2.5

diffuse_gamma = 2.19

# Conversion to 1GeV

diffuse_flux *= (4 * np.pi * u.sr) * (10 ** 5) ** diffuse_gamma

# diffuse_flux *= (4 * np.pi * u.sr) * (
#         code_ref_flux/diffuse_ref_flux.to(code_ref_flux.unit))**-diffuse_gamma


diffuse_fluence = diffuse_flux.to("GeV-1 cm-2 yr-1") * (1./12.) * u.yr

# print "Time-Integrated Diffuse Flux in 1 month:", diffuse_fluence

def sfr_madau(z):
    """
    star formation history
    http://arxiv.org/pdf/1403.0007v3.pdf
    Madau & Dickinson 2014, Equation 15
    result is in solar masses/year/Mpc^3



    """
    rate = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) /(
        u.Mpc**3 * u.year
    )

    return rate


def ccsn_madau(z):
    """"Best fit k from same paper"""
    return 0.0068 * sfr_madau(z)


def sfr_clash_candels(z):
    """
    star formation history
    https://arxiv.org/pdf/1509.06574.pdf

    result is in solar masses/year/Mpc^3

    Can match Figure 6, if the h^3 factor is divided out
    """
    rate = 0.015 * cosmo.h**3 * (1+z)**5.0 / (1 + ((1+z)/1.5)**6.1) /(
        u.Mpc**3 * u.year
    )

    return rate


def ccsn_clash_candels(z):
    """Best fit k from paper"""
    return 0.0091 * cosmo.h**2 * sfr_clash_candels(z)


def integrate_over_z(f, zmin=0.0, zmax=8.0):

    nsteps = 1e3

    zrange, step = np.linspace(zmin, zmax, nsteps + 1, retstep=True)
    int_sum = 0.0

    for i, z in enumerate(zrange[1:-1]):
        int_sum += 0.5 * step * (f(z) + f(zrange[i+1]))

    return int_sum


def cumulative_z(f, zrange):

    ints = []

    nsteps = 1e3 + 1

    if isinstance(zrange, np.ndarray):
        step = zrange[1] - zrange[0]
    else:
        zrange, step = np.linspace(0.0, zrange, nsteps + 1, retstep=True)

    int_sum = 0.0

    for i, z in enumerate(zrange[1:-1]):
        int_sum += 0.5 * step * (f(z) + f(zrange[i + 1]))
        ints.append(astropy.units.quantity.Quantity(int_sum))

    return ints


def define_cosmology_functions(rate, nu_e, gamma, nu_bright_fraction):

    def rate_per_z(z):
        """ Equals rate as a function of z, multiplied by the differential
        comoving volume, multiplied by 4pi steradians for full sphere,
        multiplied by the neutrino-bright fraction, and then divided by (1+z)
        to account for time dilation which reduces the rate of transients at
        high redshifts."""
        return rate(z) * cosmo.differential_comoving_volume(z) * \
               nu_bright_fraction * (4 * np.pi * u.sr)/(1+z)

    def nu_flux_per_z(z):
        return rate_per_z(z).to("s-1") * nu_e * (1+z)**(4-gamma) / (
                4 * np.pi * Distance(z=z).to("cm")**2)

    cumulative_nu_flux = lambda z: cumulative_z(nu_flux_per_z, z)

    return rate_per_z, nu_flux_per_z, cumulative_nu_flux


def calculate_transient(nu_e, rate, name, zmax=8., gamma=diffuse_gamma,
                        nu_bright_fraction=1.0,
                        diffuse_fraction=None):

    print "Diffuse Flux at 100 TeV:", diffuse_flux
    print "Diffuse Flux at 1 GeV:", "{:.3E}".format(diffuse_flux)
    print "Diffuse Spectral Index is", diffuse_gamma

    # nu_e *= 3

    print "\n"
    print name
    print "\n"
    print "Neutrino Energy is", nu_e
    print "Rate is", rate(0.0)

    savedir = plots_dir + "cosmology/" + name + "/"

    try:
        os.makedirs(savedir)
    except OSError:
        pass

    fluence_conversion = fluence_integral(gamma)

    nu_e = nu_e.to("GeV") / fluence_conversion

    print "Flux at 1GeV is", nu_e

    zrange, step = np.linspace(0.0, zmax, 1 + 1e3, retstep=True)

    rate_per_z, nu_flux_per_z, cumulative_nu_flux = define_cosmology_functions(
        rate, nu_e, gamma, nu_bright_fraction
    )

    nu_at_horizon = cumulative_nu_flux(8)[-1]

    print "Local rate at z=0.0", "{:.3E}".format(rate(0.0))
    print "Local rate at z=1.0", "{:.3E}".format(rate(1.0))
    print "Local rate at z=1.4", "{:.3E}".format(rate(1.4))
    print "Cumulative sources at z=8.0:",
    print "{:.3E}".format(cumulative_z(rate_per_z, 8.0)[-1].value)
    print "Cumulative flux at z=8.0:", "{:.3E}".format(nu_at_horizon)
    print "Neutrino diffuse flux:", "{:.3E}".format(diffuse_flux)
    ratio = nu_at_horizon.value / diffuse_flux.value
    print "Fraction of diffuse flux", ratio

    if diffuse_fraction is not None:
        print "Scaling flux so that, at z=8, the contribution is equal to", \
            diffuse_fraction
        nu_e *= diffuse_fraction / ratio
        print "Neutrino Energy rescaled to", \
            (nu_e * fluence_conversion).to("erg")

    plt.figure()
    plt.plot(zrange, rate(zrange))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.savefig(savedir + 'rate.pdf')
    plt.close()

    print "Sanity Check:"
    print "Integrated Source Counts \n"

    for z in [0.01, 0.08, 0.1, 0.2, 0.3]:
        print z, Distance(z=z).to("Mpc"), cumulative_z(rate_per_z, z)[-1]

    print "Fraction from nearby sources", cumulative_nu_flux(0.3)[-1]/ \
                                          nu_at_horizon

    plt.figure()
    plt.plot(zrange[1:-1], [x.value for x in cumulative_z(rate_per_z, zrange)])
    plt.yscale("log")
    plt.ylabel("Cumulative Sources")
    plt.xlabel("Redshift")
    plt.savefig(savedir + 'integrated_source_count.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange, nu_flux_per_z(zrange))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.savefig(savedir + 'diff_vol_contribution.pdf')
    plt.close()

    cum_nu = [x.value for x in cumulative_nu_flux(zrange)]

    plt.figure()
    plt.plot(zrange[1:-1], cum_nu)
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(r"Cumulative Neutrino Flux [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]")
    plt.axhline(y=diffuse_flux.value, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(savedir + 'int_nu_flux_contribution.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange[1:-1],
             [nu_flux_per_z(z).value for z in zrange[1:-1]])
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(
        r"Differential Neutrino Flux [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ dz]")
    plt.axhline(y=diffuse_flux.value, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(savedir + 'diff_nu_flux_contribution.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange[1:-1],
             [(nu_e / (4 * np.pi * Distance(z=z).to("cm")**2)).value
              for z in zrange[1:-1]])
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(
        r"Time-Integrated Flux per Source [ GeV$^{-1}$ cm$^{-2}$]")
    plt.axhline(y=diffuse_fluence.value, color="red", linestyle="--",
                label="1 month Time-Integrated Diffuse Flux")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savedir + 'nu_flux_per_source_contribution.pdf')
    plt.close()

    # print "\n"
    #
    # print "Let's assume that 50% of the diffuse flux is distributed on a \n" \
    #       "single source in the northern sky. That's unrealistic, but \n " \
    #       "should give us an idea of expected neutrino numbers! \n"
    #
    # source = np.load(ps_catalogue_name(0.2))
    #
    # inj_kwargs = {
    #     "Injection Time PDF": {
    #         "Name": "Steady"
    #     },
    #     "Injection Energy PDF": {
    #         "Name": "Power Law",
    #         "Gamma": diffuse_gamma,
    #         "Flux at 1GeV": diffuse_flux*0.5
    #     }
    # }
    #
    # calculate_neutrinos_from_flux(source[0], gfu_dict, inj_kwargs)


results = [
    ["SNIIn", 1.5 * 10**49 * u.erg, 0.064],
    ["SNIIP", 6 * 10**48 * u.erg, 0.52],
    ["SN1bc", 4.5 * 10**48 * u.erg, 0.069 + 0.176]
]

# for [name, nu_e, fraction] in results:
#
#     def f(z):
#         return fraction * ccsn_clash_candels(z)
#
#     calculate_transient(nu_e, f, name, zmax=0.3)

if __name__ == "__main__":
    calculate_transient(1 * 10**49 * u.erg, ccsn_clash_candels, "CCSN",
                        zmax=2.5)


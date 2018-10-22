from astropy import units as u
import astropy
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
import numpy as np
import os
from flarestack.shared import plots_dir
from flarestack.utils.neutrino_astronomy import fluence_integral

# IceCube Diffuse Flux @ 100TeV

diffuse_flux = 0.90 * 10**-18 * u.GeV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1

print "Diffuse Flux at 100 TeV:", diffuse_flux

# Best Fit Spectral Index

diffuse_gamma = 2.13

# Conversion to 1GeV

diffuse_flux *= (4 * np.pi * u.sr) * (10**5) ** diffuse_gamma

print "Diffuse Flux at 1 GeV:", diffuse_flux

diffuse_fluence = diffuse_flux.to("GeV-1 cm-2 yr-1") * (1./12.) * u.yr

print "Time-Integrated Diffuse Flux in 1 month:", diffuse_fluence

nu_e = 10**50 * u.erg

savedir = plots_dir + "cosmology/"

try:
    os.makedirs(savedir)
except OSError:
    pass

def sfr_madau(z):
    """
    star formation history
    http://arxiv.org/pdf/1403.0007v3.pdf
    Madau & Dickinson 2014, Equation 15
    result is in solar masses/year/Mpc^3, assume this is proportional to number of sources
    """
    rate = 2 * 10**-6 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) /(
        u.Mpc**3 * u.year
    )

    return rate


def integrate_over_z(f, zmin=0.0, zmax=8.0):

    nsteps = 1e3

    zrange, step = np.linspace(zmin, zmax, nsteps + 1, retstep=True)
    int_sum = 0.0

    for i, z in enumerate(zrange[1:-1]):
        int_sum += 0.5 * step * (f(z) + f(zrange[i+1]))

    return int_sum

def calculate_transient(nu_e, rate, zmax=8., gamma=diffuse_gamma,
                        nu_bright_fraction=1.0,
                        diffuse_fraction=None):
    print "Neutrino Energy is", nu_e
    print "Rate is", rate(0.0)

    fluence_conversion = fluence_integral(gamma)

    nu_e = nu_e.to("GeV") / fluence_conversion

    zrange, step = np.linspace(0.0, zmax, 1 + 1e3, retstep=True)

    def rate_per_z(z):
        return rate(z) * cosmo.differential_comoving_volume(z) * \
               nu_bright_fraction * (4 * np.pi * u.sr)/(1 + z)

    def nu_flux_per_z(z):
        return rate_per_z(z).to("s-1") * nu_e / (
                4 * np.pi * Distance(z=z).to("cm")**2)

    def cumulative_nu_flux(zrange):
        ints = []

        nsteps = 1e3 + 1

        if isinstance(zrange, np.ndarray):
            step = zrange[1] - zrange[0]
        else:
            zrange, step = np.linspace(0.0, zrange, nsteps + 1, retstep=True)

        int_sum = 0.0

        for i, z in enumerate(zrange[1:-1]):
            int_sum += 0.5 * step * (nu_flux_per_z(z) +
                                     nu_flux_per_z(zrange[i + 1]))
            ints.append(astropy.units.quantity.Quantity(int_sum))

        return ints

    nu_at_horizon = cumulative_nu_flux(8)[-1]

    print "Cumulative flux at z=8.0:", nu_at_horizon
    print "Neutrino diffuse flux:", diffuse_flux
    ratio = nu_at_horizon.value / diffuse_flux.value
    print "Fraction of diffuse flux", ratio

    if diffuse_fraction is not None:
        print "Scaling flux so that, at z=8, the contribution is equal to", \
            diffuse_fraction
        nu_e *= diffuse_fraction / ratio
        print "Neutrino Energy rescaled to", \
            (nu_e * fluence_conversion).to("erg")

    plt.figure()
    plt.plot(zrange, sfr_madau(zrange))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.savefig(savedir + 'sfr_madau.pdf')
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
             [nu_flux_per_z(z).value * step for z in zrange[1:-1]])
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


calculate_transient(nu_e, sfr_madau, zmax=8., diffuse_fraction=1.0)


import logging
from astropy import units as u
import astropy
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
import numpy as np
import os
from flarestack.shared import plots_dir
from flarestack.core.energy_pdf import EnergyPDF, read_e_pdf_dict
from flarestack.cosmo.icecube_diffuse_flux import get_diffuse_flux_at_1GeV

logger = logging.getLogger(__name__)


def integrate_over_z(f, zmin=0.0, zmax=8.0):

    nsteps = 1e3

    zrange, step = np.linspace(zmin, zmax, int(nsteps + 1), retstep=True)
    int_sum = 0.0

    for i, z in enumerate(zrange[1:-1]):
        int_sum += 0.5 * step * (f(z) + f(zrange[i+2]))

    return int_sum


def cumulative_z(f, zrange):

    ints = []

    nsteps = 1e3 + 1

    if isinstance(zrange, np.ndarray):
        step = zrange[1] - zrange[0]
    else:
        zrange, step = np.linspace(0.0, zrange, int(nsteps + 1), retstep=True)

    int_sum = 0.0

    for i, z in enumerate(zrange[1:-1]):
        int_sum += 0.5 * step * (f(z) + f(zrange[i + 2]))
        ints.append(astropy.units.quantity.Quantity(int_sum))

    return ints


def define_cosmology_functions(rate, nu_e_flux_1GeV, gamma,
                               nu_bright_fraction=1.):

    def rate_per_z(z):
        """ Equals rate as a function of z, multiplied by the differential
        comoving volume, multiplied by 4pi steradians for full sphere,
        multiplied by the neutrino-bright fraction, and then divided by (1+z)
        to account for time dilation which reduces the rate of transients at
        high redshifts.

        :param z: Redshift
        :return: Transient rate in shell at that redshift
        """
        return rate(z) * cosmo.differential_comoving_volume(z) * \
               nu_bright_fraction * (4 * np.pi * u.sr) / (1+z)

    def nu_flux_per_source(z):
        """Calculate the time-integrated neutrino flux contribution on Earth
        per source. Equal to the flux normalisation per source at 1GeV,
        divided by the sphere 4 pi dl^2 to give the flux at 1GeV on Earth.
        This then needs to be corrected by factors of (1+z)-gamma to account
        for the redshifting of the spectrum to lower energy values. This
        assumes that he power law extends beyond the traditional icecube
        sensitivity range.

        :param z: Redshift of shell
        :return: Neutrino flux from shell at Earth
        """
        return nu_e_flux_1GeV * (1 + z) ** (3 - gamma) / (
                4 * np.pi * Distance(z=z).to("cm")**2)

    def nu_flux_per_z(z):
        """Calculate the neutrino flux contribution on Earth that each
        redshift shell contributes. Equal to the rate of sources per shell,
        multiplied by the flux normalisation per source at 1GeV, divided by
        the sphere 4 pi dl^2 to give the flux at 1GeV on Earth. This then
        needs to be corrected by factors of (1+z)-gamma to account for the
        redshifting of the spectrum to lower energy values. This assumes that
        the power law extends beyond the traditional icecube sensitivity range.

        :param z: Redshift of shell
        :return: Neutrino flux from shell at Earth
        """
        return rate_per_z(z).to("s-1") * nu_flux_per_source(z) / (
                4 * np.pi * u.sr)

    def cumulative_nu_flux(z):
        """Calculates the integrated neutrino flux on Earth for all sources
        lying within a sphere up to the given redshift. Uses numerical
        intergration to calculate this, given the source rate and neutrino
        flux per source.

        :param z: Redshift up to which neutrino flux is integrated
        :return: Cumulative neutrino flux at 1 GeV
        """
        return cumulative_z(nu_flux_per_z, z)

    return rate_per_z, nu_flux_per_z, nu_flux_per_source, cumulative_nu_flux


def calculate_transient_cosmology(e_pdf_dict, rate, name, zmax=8.,
                                  nu_bright_fraction=1.0,
                                  diffuse_fraction=None,
                                  diffuse_fit="joint_15"):

    e_pdf_dict = read_e_pdf_dict(e_pdf_dict)

    diffuse_flux, diffuse_gamma = get_diffuse_flux_at_1GeV(diffuse_fit)

    logger.info("Using the {0} best fit values of the diffuse flux.".format(diffuse_fit))
    # print "Raw Diffuse Flux at 1 GeV:", diffuse_flux / (4 * np.pi * u.sr)
    logger.info("Diffuse Flux at 1 GeV: {0}".format(diffuse_flux))
    logger.info("Diffuse Spectral Index is {0}".format(diffuse_gamma))

    if "gamma" not in e_pdf_dict:
        logger.warning("No spectral index has been specified. "
                        "Assuming source has spectral index matching diffuse flux")
        e_pdf_dict["gamma"] = diffuse_gamma

    savedir = plots_dir + "cosmology/" + name + "/"

    try:
        os.makedirs(savedir)
    except OSError:
        pass

    energy_pdf = EnergyPDF.create(e_pdf_dict)
    fluence_conversion = energy_pdf.fluence_integral() * u.GeV ** 2

    if "nu_flux_at_1_gev" in e_pdf_dict.keys():
        nu_flux_at_1_gev = e_pdf_dict["nu_flux_at_1_gev"].to("GeV-1")
        nu_e = (nu_flux_at_1_gev * fluence_conversion).to("erg")

    else:
        nu_e = e_pdf_dict["source_energy_erg"]
        nu_flux_at_1_gev = nu_e.to("GeV") / fluence_conversion

    gamma = e_pdf_dict["gamma"]

    logger.info(f"Neutrino Energy is {nu_e:.2g}")
    logger.info(f"Neutrino Flux at 1 GeV is {nu_flux_at_1_gev:.2g}")
    logger.info(f"Local rate is {rate(0.0):.2g}")

    zrange, step = np.linspace(0.0, zmax, int(1 + 1e3), retstep=True)

    rate_per_z, nu_flux_per_z, nu_flux_per_source, cumulative_nu_flux = \
        define_cosmology_functions(rate, nu_flux_at_1_gev, gamma, nu_bright_fraction)

    logger.info(f"Cumulative sources at z=8.0: {cumulative_z(rate_per_z, 8.0)[-1].value:.2g}")

    nu_at_horizon = cumulative_nu_flux(8)[-1]

    logger.info(f"Cumulative flux at z=8.0 (1 GeV): {nu_at_horizon:.2g}")
    logger.info(f"Cumulative annual flux at z=8.0 (1 GeV): {(nu_at_horizon * u.yr).to('GeV-1 cm-2 sr-1'):.2g}")

    ratio = nu_at_horizon.value / diffuse_flux.value
    logger.info(f"Fraction of diffuse flux at 1GeV: {ratio:.2g}")
    logger.info(f"Cumulative neutrino flux {nu_at_horizon:.2g}")
    logger.debug(f"Diffuse neutrino flux {diffuse_flux:.2g}")

    if diffuse_fraction is not None:
        logger.info(f"Scaling flux so that, at z=8, the contribution is equal to {diffuse_fraction:.2g}")
        nu_flux_at_1_gev *= diffuse_fraction / ratio
        logger.info(f"Neutrino Energy rescaled to {(nu_flux_at_1_gev * fluence_conversion).to('erg'):.2g}")

    plt.figure()
    plt.plot(zrange, rate(zrange))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(r"Rate [Mpc$^{-3}$ year$^{-1}$]")
    plt.tight_layout()
    plt.savefig(savedir + 'rate.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange, rate_per_z(zrange) / rate(zrange))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(r"Differential Comoving Volume [Mpc$^{3}$ dz]")
    plt.tight_layout()
    plt.savefig(savedir + 'comoving_volume.pdf')
    plt.close()

    logger.debug("Sanity Check:")
    logger.debug("Integrated Source Counts \n")

    for z in [0.01, 0.08, 0.1, 0.2, 0.3, 0.7,  8]:
        logger.debug("{0}, {1}, {2}".format(
            z, Distance(z=z).to("Mpc"), cumulative_z(rate_per_z, z)[-1])
        )

    for nearby in [0.1, 0.3]:

        logger.info(
            f"Fraction of flux from nearby (z<{nearby}) sources: {cumulative_nu_flux(nearby)[-1] / nu_at_horizon:.2g}")

    plt.figure()
    plt.plot(zrange, rate_per_z(zrange))
    plt.yscale("log")
    plt.ylabel("Differential Source Rate [year$^{-1}$ dz]")
    plt.xlabel("Redshift")
    plt.tight_layout()
    plt.savefig(savedir + 'diff_source_count.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange[1:-1], [x.value for x in cumulative_z(rate_per_z, zrange)])
    plt.yscale("log")
    plt.ylabel("Cumulative Sources")
    plt.xlabel("Redshift")
    plt.tight_layout()
    plt.savefig(savedir + 'integrated_source_count.pdf')
    plt.close()

    plt.figure()
    plt.plot(zrange[1:-1], nu_flux_per_z(zrange[1:-1]))
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.tight_layout()
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
             [(nu_flux_per_source(z)).value for z in zrange[1:-1]])
    plt.yscale("log")
    plt.xlabel("Redshift")
    plt.ylabel(
        r"Time-Integrated Flux per Source [ GeV$^{-1}$ cm$^{-2}$]")
    plt.tight_layout()
    plt.savefig(savedir + 'nu_flux_per_source_contribution.pdf')
    plt.close()

    return nu_at_horizon

import logging
import matplotlib.pyplot as plt
from astropy import units as u
from flarestack.shared import illustration_dir
from flarestack.cosmo.icecube_diffuse_flux.joint_15 import joint_15
from flarestack.cosmo.icecube_diffuse_flux.nt_16 import nt_16
from flarestack.cosmo.icecube_diffuse_flux.nt_17 import nt_17
from flarestack.cosmo.icecube_diffuse_flux.nt_19 import nt_19

contours = {**joint_15, **nt_16, **nt_17, **nt_19}

def get_diffuse_flux_at_100TeV(fit="joint_15"):
    """Returns value for the diffuse neutrino flux, based on IceCube's data.
    The fit can be specified (either 'Joint' or 'Northern Tracks') to get
    corresponding values from different analyses

    :param fit: Fit of diffuse flux to be used
    :return: Best fit diffuse flux at 100 TeV, and best fit spectral index
    """

    if fit == "joint":
        logging.warning("Fit 'joint' was used, without a specified year. "
                        "Assuming 'joint_15', from https://arxiv.org/abs/1507.03991.")
        fit = "joint_15"

    if fit == "northern_tracks":
        logging.warning("Fit 'northern_tracks' was used, without a specified year. "
                        "Assuming 'northern_tracks_19', from https://arxiv.org/abs/1908.09551.")
        fit = "northern_tracks_19"

    if fit in contours.keys():
        diffuse_flux, diffuse_gamma, _, _, _ = contours[fit]

    else:
        raise Exception(f"Fit '{fit}' not recognised! \n The following fits are available: \n {contours.keys()}")

    return diffuse_flux, diffuse_gamma

def get_diffuse_flux_at_1GeV(fit="joint_15"):
    """Returns the IceCube diffuse flux at 1GeV, to match flarestack
    convention for flux measurements.

    :param fit: Fit of diffuse flux to be used
    :return: Best fit diffuse flux at 1 GeV, and best fit spectral index
    """
    diffuse_flux, diffuse_gamma = get_diffuse_flux_at_100TeV(fit)
    return diffuse_flux * (10 ** 5) ** diffuse_gamma, diffuse_gamma


def upper_contour(energy_range, contour):
    """Trace upper contour"""
    return [max([f(energy, norm, index) for (index, norm) in contour])
            for energy in energy_range]


def lower_contour(energy_range, contour):
    """Trace lower contour"""
    return [min([f(energy, norm, index) for (index, norm) in contour])
            for energy in energy_range]


def f(energy, norm, index):
    return norm * energy ** -index * (10**5) ** index

def plot_diffuse_flux(label, contour_68, contour_95, e_range):


    plt.figure()

    # Plot 68% contour

    for (index, norm) in contour_68:
        plt.plot(e_range,
                 e_range ** 2 * f(e_range, norm, index),
                 alpha=0.6)
    plt.plot(
        e_range,
        e_range ** 2 * upper_contour(e_range, contour_68),
        color="k", label="68% contour", linestyle="-"
    )
    plt.plot(
        e_range,
        e_range ** 2 * lower_contour(e_range, contour_68),
        color="k", linestyle="-",
    )

    # Plot 95% contour

    for (index, norm) in contour_95:
        plt.plot(e_range,
                 e_range ** 2 * f(e_range, norm, index),
                 alpha=0.3)
    plt.plot(
        e_range,
        e_range ** 2 * upper_contour(e_range, contour_95),
        color="k", linestyle="--", label="95% contour"
    )
    plt.plot(
        e_range,
        e_range ** 2 * lower_contour(e_range, contour_95),
        color="k", linestyle="--"
    )

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title(r"Diffuse Flux Global Best Fit ($\nu_{\mu} + \bar{\nu}_{\mu})$")
    plt.ylabel(r"$E^{2}\frac{dN}{dE}$[GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    plt.xlabel(r"$E_{\nu}$ [GeV]")
    savepath = illustration_dir + "diffuse_flux_global_fit.pdf"
    plt.savefig(savepath)
    plt.close()
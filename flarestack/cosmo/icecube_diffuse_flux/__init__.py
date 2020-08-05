import logging
import matplotlib.pyplot as plt
from flarestack.shared import illustration_dir
from flarestack.cosmo.icecube_diffuse_flux.joint_15 import joint_15
from flarestack.cosmo.icecube_diffuse_flux.nt_16 import nt_16
from flarestack.cosmo.icecube_diffuse_flux.nt_17 import nt_17
from flarestack.cosmo.icecube_diffuse_flux.nt_19 import nt_19

logger = logging.getLogger(__name__)

contours = {**joint_15, **nt_16, **nt_17, **nt_19}

def load_fit(fit):
    if fit == "joint":
        logger.warning("Fit 'joint' was used, without a specified year. "
                        "Assuming 'joint_15', from https://arxiv.org/abs/1507.03991.")
        fit = "joint_15"

    if fit == "northern_tracks":
        logger.warning("Fit 'northern_tracks' was used, without a specified year. "
                        "Assuming 'northern_tracks_19', from https://arxiv.org/abs/1908.09551.")
        fit = "northern_tracks_19"

    if fit not in contours.keys():
        raise Exception(f"Fit '{fit}' not recognised! \n The following fits are available: \n {contours.keys()}")

    best_fit_flux, best_fit_gamma, contour_68, contour_95, e_range, ref = contours[fit]

    logger.info(f"Loaded contour '{fit}' from {ref}")

    return best_fit_flux, best_fit_gamma, contour_68, contour_95, e_range


def get_diffuse_flux_at_100TeV(fit="joint_15"):
    """Returns value for the diffuse neutrino flux, based on IceCube's data.
    The fit can be specified (either 'Joint' or 'Northern Tracks') to get
    corresponding values from different analyses

    :param fit: Fit of diffuse flux to be used
    :return: Best fit diffuse flux at 100 TeV, and best fit spectral index
    """

    diffuse_flux, diffuse_gamma, _, _, _ = load_fit(fit)

    return diffuse_flux, diffuse_gamma

def get_diffuse_flux_at_1GeV(fit="joint_15"):
    """Returns the IceCube diffuse flux at 1GeV, to match flarestack
    convention for flux measurements.

    :param fit: Fit of diffuse flux to be used
    :return: Best fit diffuse flux at 1 GeV, and best fit spectral index
    """
    diffuse_flux, diffuse_gamma = get_diffuse_flux_at_100TeV(fit)
    return diffuse_flux * (10 ** 5) ** diffuse_gamma, diffuse_gamma

def get_diffuse_flux(e_gev=1., fit="joint_15"):
    """Returns the IceCube diffuse flux at a given energy.

    :param e_gev: Energy to evaluate flux at
    :param fit: Fit of diffuse flux to be used
    :return: Best fit diffuse flux at 1 GeV, and best fit spectral index
    """
    diffuse_flux, diffuse_gamma = get_diffuse_flux_at_1GeV(fit)
    return diffuse_flux * e_gev ** -diffuse_gamma, diffuse_gamma

def flux_f(energy, norm, index):
    """Flux function

    :param energy: Energy to evaluate
    :param norm: Flux normalisation at 100 TeV
    :param index: Spectral index
    :return: Flux at given energy
    """
    return norm * (energy ** -index) * (10.**5) ** index

def upper_contour(energy_range, contour):
    """Trace upper contour"""
    return [max([flux_f(energy, norm, index) for (index, norm) in contour])
            for energy in energy_range]

def lower_contour(energy_range, contour):
    """Trace lower contour"""
    return [min([flux_f(energy, norm, index) for (index, norm) in contour])
            for energy in energy_range]

def get_diffuse_flux_contour(fit="joint_15", contour_name="68"):
    """Provides functions to generate 'butterfly plot' of diffuse flux contours.

    :param fit: Diffuse Flux fit to use
    :param contour_name: Contour to provide ('68' or '95')
    :return: Best-fit function, upper butterfly function, lower butterfly function, energy range
    """

    best_fit_flux, best_fit_gamma, contour_68, contour_95, e_range = load_fit(fit)

    if contour_name in [68., 68, 0.68, "68", "0.68", "68%"]:
        contour = contour_68
    elif contour_name in [95., 95, 0.95, "95", "0.95", "95%"]:
        contour = contour_95
    else:
        raise Exception(f"Contour '{contour_name}' not recognised. Please select '68' or '95'.")

    upper_f = lambda e: upper_contour(e, contour)
    lower_f = lambda e: lower_contour(e, contour)
    best_f = lambda e: flux_f(e, best_fit_flux, best_fit_gamma)

    return best_f, upper_f, lower_f, e_range

def plot_diffuse_flux(fit="joint_15"):
    plt.figure()

    best_fit_flux, best_fit_gamma, contour_68, contour_95, e_range = load_fit(fit)

    for i, contour in enumerate([68., 95.]):
        all_c = [contour_68, contour_95][i]

        best_f, upper_f, lower_f, e_range = get_diffuse_flux_contour(fit=fit, contour_name=contour)

        plt.plot(e_range,
                 e_range ** 2 * best_f(e_range),
                 alpha=1.0, color="k")

        plt.plot(e_range,
                 e_range ** 2 * lower_f(e_range),
                 color=f"C{i}",
                 label=f"{contour} %",
                 )

        plt.plot(e_range,
                 e_range ** 2 * upper_f(e_range),
                 color=f"C{i}",
                 )

        for (index, norm) in all_c:

            plt.plot(e_range,
                     e_range**2 * flux_f(e_range, norm, index),
                     alpha=0.15
                     )

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title(r"Diffuse Flux Global Best Fit ($\nu_{\mu} + \bar{\nu}_{\mu})$")
    plt.ylabel(r"$E^{2}\frac{dN}{dE}$[GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    plt.xlabel(r"$E_{\nu}$ [GeV]")
    plt.tight_layout()
    savepath = f"{illustration_dir}diffuse_flux_global_fit_{fit}.pdf"

    logger.info(f"Saving to {savepath}")

    plt.savefig(savepath)
    plt.close()
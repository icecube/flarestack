from __future__ import division
from builtins import str
from flarestack.utils.neutrino_cosmology import calculate_transient_cosmology, \
    sfr_madau, sfr_clash_candels, get_diffuse_flux_at_1GeV
from flarestack.analyses.ccsn.stasik_2017.ccsn_limits import limits
from flarestack.core.energy_pdf import EnergyPDF
from astropy.cosmology import Planck15 as cosmo
import numpy as np
from flarestack.cosmo.icecube_diffuse_flux.joint_15 import contour_95, e_range
from flarestack.cosmo import lower_contour, upper_contour
import matplotlib.pyplot as plt
from flarestack.shared import plot_output_dir
import os
from flarestack.cosmo.rates import ccsn_clash_candels, ccsn_madau, get_sn_fraction, get_sn_type_rate

global_fit_e_range = e_range

if __name__ == "__main__":

    e_pdf_dict_template = {
        "energy_pdf_name": "power_law",
        "e_min_gev": 10 ** 2,
        "e_max_gev": 10 ** 7,
    }

    results = [
        ["IIn", 1.0],
        # ["IIp", 1.0],
        # ["Ibc", 1.0]
    ]

    norms = dict()

    for [name, nu_bright] in results:

        def f(x):
            return get_sn_type_rate(sn_type=name)(x) * nu_bright

        e_pdf_dict = dict(e_pdf_dict_template)

        energy_pdf = EnergyPDF.create(e_pdf_dict)

        e_pdf_dict["Source Energy (erg)"] = limits[name]["Fixed Energy (erg)"]
        # e_pdf_dict["Source Energy (erg)"] = ccsn_energy_limit(name,
        # diffuse_gamma)
        norms[name] = calculate_transient_cosmology(e_pdf_dict, f, name, zmax=6.0,
                                                    nu_bright_fraction=nu_bright,
                                                    diffuse_fit="joint")

    base_dir = plot_output_dir("analyses/ccsn/")

    e_range = np.logspace(2.73, 5.64, 3)

    try:
        os.makedirs(base_dir)
    except OSError:
        pass

    def z(energy, norm):
        return norm * energy ** -0.5

    plt.figure()

    # Plot 95% contour

    plt.fill_between(
        global_fit_e_range,
        global_fit_e_range ** 2 * upper_contour(global_fit_e_range, contour_95),
        global_fit_e_range ** 2 * lower_contour(global_fit_e_range, contour_95),
        color="k", label='IceCube diffuse flux\nApJ 809, 2015',
        alpha=.5,
    )

    diffuse_norm, diffuse_gamma = get_diffuse_flux_at_1GeV("joint")

    plt.plot(global_fit_e_range,
             diffuse_norm * global_fit_e_range ** (2. - diffuse_gamma),
             color="k")

    for i,(name, norm) in enumerate(norms.items()):
        # plt.plot(e_range, z(e_range, norm), label=name)
        plt.errorbar(e_range, z(e_range, norm).value,
                     yerr=.25 * np.array([x.value for x in z(e_range, norm)]),
                     uplims=True, color=["b", "r", "orange"][i],
                     label="Supernovae Type {0}".format(name))

    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title(r"Diffuse Flux Global Best Fit ($\nu_{\mu} + \bar{\nu}_{\mu})$")
    plt.ylabel(r"$E^{2}\frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    plt.xlabel(r"$E_{\nu}$ [GeV]")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(base_dir + "diffuse_flux_global_fit.pdf")
    plt.close()


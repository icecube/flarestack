from flarestack.utils.neutrino_cosmology import calculate_transient
from astropy import units as u
import matplotlib.pyplot as plt
from flarestack.shared import plot_output_dir
import numpy as np
import os
from flarestack.misc.convert_diffuse_flux_contour import contour_95, \
    upper_contour, lower_contour, global_fit_e_range
from flarestack.utils.neutrino_cosmology import get_diffuse_flux_at_1GeV
from flarestack.analyses.tde.shared_TDE import tde_cat_limit, \
    tde_cat_weight_limit

e_pdf_dict_template = {
    "Name": "Power Law",
    # "Source Energy (erg)": 1.0 * 10**54 * u.erg,
    "E Min": 10**2,
    "E Max": 10**7,
    # "Gamma": 2.0
}

# Assumed source evolution is highly negative
non_jetted_m = 0

eta = -2


def raw_tde_evolution(z):
    """https://arxiv.org/abs/1509.01592"""
    evolution = ((1 + z)**(0.2 * eta) + ((1 + z)/1.43)**(-3.2 * eta) +
                 ((1 + z)/2.66)**(-7 * eta)
                 )**(1./eta)
    return evolution


def normed_tde_evolution(z):
    """TDE evolution, scaled such that normed_evolution(0) = 1. This function
    can then be multiplied by the local TDE density to give overall TDE
    redshift evolution

    :param z: Redshift
    :return: Normalised scaling of density with redshift
    """
    return raw_tde_evolution(z)/raw_tde_evolution(0.0)



def tde_rate(z):
    """TDE rate as a function of redshift, equal to the local density
    multiplied by the redshift evolution of the density. Rate is derived
    from https://arxiv.org/pdf/1707.03458

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return normed_tde_evolution(z) * 8 * 10**-7 / (u.Mpc**3 * u.yr)



def theoretical_tde_rate(z):
    """Optimistic TDE rate from https://arxiv.org/pdf/1601.06787 as a function
    of redshift, equal to the local density multiplied by the redshift
    evolution of the density

    :param z: Redshift
    :return: TDE rate at redshift z
    """
    return normed_tde_evolution(z) * 1.5 * 10**-6 / (u.Mpc**3 * u.yr)\


# Assumed source evolution is highly negative
jetted_m = -3


def biehl_jetted_rate(z):
    """Rate of TDEs assumed by Biehl et al. 2018 is 0.1 per Gpc per year
    (10^-10 per Mpc per year). The source evolution is assumed to be
    negative, with an index m=3, though the paper also considers indexes up
    to m=0 (flat). More details found under https://arxiv.org/abs/1711.03555

    :param z: Redshift
    :return: Jetted TDE rate
    """
    rate = 0.1 * (1 + z)**jetted_m / (u.Gpc**3 * u.yr)
    return rate.to("Mpc-3 yr-1")


def standard_jetted_rate(z):
    """Rate taken from appendix of Sun et Al (2015)
    (https://arxiv.org/pdf/1509.01592)"""
    return normed_tde_evolution(z) * 3 * 10**-11 / (u.Mpc**3 * u.yr)

jetted_rate_uncertainties = [[0.07/0.03, 0.01/0.03]]
nonjetted_rate_uncertainties = [[12./8., 4./8.]]


if __name__ == "__main__":

    all_res = dict()

    for k, limit_f in enumerate([tde_cat_weight_limit, tde_cat_limit, ]):

        res = [
            ("gold", 8 * 10**50 * u.erg, "Non-jetted TDEs"),
            # ("obscured", 4.5*10**51 * u.erg),
            # ("silver", 3 * 10**49 * u.erg)
        ]

        norms = dict()
        lims = dict()

        for (name, energy, key) in res:

            class_dict = dict()

            lim_dict = dict()

            # for i, rate in enumerate([tde_rate, theoretical_tde_rate]):
            for i, rate in enumerate([tde_rate]):
                e_pdf_dict = dict(e_pdf_dict_template)
                e_pdf_dict["Source Energy (erg)"] = limit_f(name, 2.5) * u.erg

                rate_key = ["(Observed Rate)", "(Theoretical Rate)"][i]

                class_dict[rate_key] = calculate_transient(e_pdf_dict, rate,
                                                      name + "TDEs",
                                                      zmax=6.0, diffuse_fit="Joint")

                lim_dict[rate_key] = nonjetted_rate_uncertanties[i]

            norms[key] = class_dict
            lims[key] = lim_dict

        norms["Jetted TDEs"] = dict()
        lims["Jetted TDEs"] = dict()

        e_pdf_dict = dict(e_pdf_dict_template)
        e_pdf_dict["Source Energy (erg)"] = limit_f("jetted", 2.5) * u.erg
        # for i, rate in enumerate([standard_jetted_rate, biehl_jetted_rate]):
        for i, rate in enumerate([standard_jetted_rate]):
            rate_key = ["(Observed Rate)", "(Theoretical Rate)"][i]
            norms["Jetted TDEs"][rate_key] = calculate_transient(e_pdf_dict,
                                                       rate,
                                                       "jetted TDEs", zmax=2.5,
                                                       diffuse_fit="Joint")

            lims["Jetted TDEs"][rate_key] = jetted_rate_uncertainties[i]

        label = ["mass_weighted", "standard_candle"][k]

        print norms, label

        all_res[label] = (norms, lims)

    # for (label, (norms, lims)) in all_res.iteritems():
    #
    #     base_dir = plot_output_dir("analyses/tde/" + label + "/")
    #
    #     e_range = np.logspace(2.73, 5.64, 3)
    #
    #     try:
    #         os.makedirs(base_dir)
    #     except OSError:
    #         pass
    #
    #
    #     def z(energy, norm):
    #         return norm * energy ** -0.5
    #
    #
    #     plt.figure()
    #     plt.subplot(111)
    #
    #     # Plot 95% contour
    #
    #     plt.fill_between(
    #         global_fit_e_range,
    #         global_fit_e_range ** 2 * upper_contour(global_fit_e_range, contour_95),
    #         global_fit_e_range ** 2 * lower_contour(global_fit_e_range, contour_95),
    #         color="k", label="IceCube\n" + r"diffuse flux$^{a}$",
    #         alpha=.5,
    #     )
    #
    #     diffuse_norm, diffuse_gamma = get_diffuse_flux_at_1GeV()
    #
    #     plt.plot(global_fit_e_range,
    #              diffuse_norm * global_fit_e_range**(2. - diffuse_gamma),
    #              color="k")
    #
    #     linestyles = {
    #         "(Observed Rate)": "-",
    #         "(Theoretical Rate)": ":"
    #     }
    #
    #     labels = [
    #         "Non-jetted TDEs\n" +
    #         r"($8^{+4}_{-4} \times 10^{-7}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
    #         "Jetted TDEs\n" +
    #         r"$(3^{+4}_{-2} \times 10^{-11}$ Mpc$^{-3}$ yr$^{-1}$)$^{c}$",
    #     ]
    #
    #     for i, (name, res_dict) in enumerate(norms.iteritems()):
    #         # plt.plot(e_range, z(e_range, norm), label=name)
    #         for (rate, norm) in res_dict.iteritems():
    #             plt.errorbar(e_range, z(e_range, norm).value,
    #                          yerr=.25 * np.array([x.value for x in z(e_range, norm)]),
    #                          uplims=True, label=labels[i],
    #                          linestyle = linestyles[rate],
    #                          color=["blue", "orange"][i])
    #
    #             [ul, ll] = lims[name][rate]
    #
    #             plt.fill_between(e_range, ll * z(e_range, norm).value,
    #                              ul * z(e_range, norm).value, alpha=0.3,
    #                              color=["blue", "orange"][i])
    #
    #     plt.yscale("log")
    #     plt.xscale("log")
    #     l  = plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.2),
    #                fancybox=True, ncol=3)
    #     for t in l.texts:
    #         t.set_multialignment('center')
    #     #plt.legend(loc="lower left")
    #     # plt.title(r"Contribution of TDEs to the Diffuse Neutrino Flux")
    #     plt.ylabel(r"$E^{2}\frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    #     plt.xlabel(r"$E_{\nu}$ [GeV]")
    #     plt.grid(True, linestyle=":")
    #     plt.tight_layout()
    #     plt.subplots_adjust(bottom=0.2, top=0.85)
    #     plt.annotate("IceCube \n Preliminary ", (0.05, 0.05), alpha=0.5, fontsize=15,
    #                  xycoords="axes fraction", multialignment="center")
    #     plt.annotate(r"With evolution from Sun et al.$^{c}$",
    #                  (0.68, -0.25),
    #                  xycoords="axes fraction", fontsize=7,
    #                  annotation_clip=False)
    #     plt.annotate("a: 2015ApJ...809...98A (IceCube Collab.)\n"
    #                  "b: 2018ApJ...852...72V (van Velzen) \n"
    #                  "c: 2015ApJ...812...33S (Sun et al.)",
    #                  (-0.15, -0.25),
    #                  xycoords="axes fraction", fontsize=7,
    #                  annotation_clip=False)
    #     plt.savefig(base_dir + "diffuse_flux_global_fit.pdf")
    #     plt.close()

    # for k, catalogue in enumerate(["Jetted TDEs"]):
    #
    #     base_dir = plot_output_dir("analyses/tde/limits_comaparison_{0}/".format(
    #         catalogue
    #     ))
    #
    #     e_range = np.logspace(2.73, 5.64, 3)
    #
    #     try:
    #         os.makedirs(base_dir)
    #     except OSError:
    #         pass
    #
    #
    #     def z(energy, norm):
    #         return norm * energy ** -0.5
    #
    #
    #     plt.figure()
    #     plt.subplot(111)
    #
    #     # Plot 95% contour
    #
    #     plt.fill_between(
    #         global_fit_e_range,
    #         global_fit_e_range ** 2 * upper_contour(global_fit_e_range,
    #                                                 contour_95),
    #         global_fit_e_range ** 2 * lower_contour(global_fit_e_range,
    #                                                 contour_95),
    #         color="k", label="IceCube\n" + r"diffuse flux$^{a}$",
    #         alpha=.5,
    #     )
    #
    #     diffuse_norm, diffuse_gamma = get_diffuse_flux_at_1GeV()
    #
    #     plt.plot(global_fit_e_range,
    #              diffuse_norm * global_fit_e_range ** (2. - diffuse_gamma),
    #              color="k")
    #
    #     linestyles = {
    #         "(Observed Rate)": "-",
    #         "(Theoretical Rate)": ":"
    #     }
    #
    #     labels = [
    #         "Jetted TDEs\n" +
    #         r"$(3^{+4}_{-2} \times 10^{-11}$ Mpc$^{-3}$ yr$^{-1}$)$^{c}$",
    #         "Non-jetted TDEs\n" +
    #         r"($8^{+4}_{-4} \times 10^{-7}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
    #     ][k]
    #
    #     for (label, (all_norms, all_lims)) in all_res.iteritems():
    #
    #         norms = all_norms[catalogue]
    #         lims = all_lims[catalogue]
    #
    #         print norms
    #
    #         for (rate, norm) in norms.iteritems():
    #
    #             plt.errorbar(e_range, z(e_range, norm.value),
    #                          yerr=.25 * np.array(
    #                              [x.value for x in z(e_range, norm.value)]),
    #                          uplims=True, label=label,
    #                          linestyle=linestyles[rate],
    #                          color=["blue", "orange"][k])
    #
    #             [ul, ll] = lims[rate]
    #
    #             plt.fill_between(e_range, ll * z(e_range, norm.value).value,
    #                              ul * z(e_range, norm.value).value, alpha=0.3,
    #                              color=["blue", "orange"][k])
    #
    #     plt.yscale("log")
    #     plt.xscale("log")
    #     l = plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.2),
    #                    fancybox=True, ncol=3)
    #     for t in l.texts:
    #         t.set_multialignment('center')
    #     # plt.legend(loc="lower left")
    #     plt.title(r"Contribution of TDEs to the Diffuse Neutrino Flux")
    #     plt.ylabel(r"$E^{2}\frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    #     plt.xlabel(r"$E_{\nu}$ [GeV]")
    #     plt.grid(True, linestyle=":")
    #     plt.tight_layout()
    #     plt.subplots_adjust(bottom=0.2, top=0.85)
    #     plt.annotate("IceCube \n Preliminary ", (0.05, 0.05), alpha=0.5,
    #                  fontsize=15,
    #                  xycoords="axes fraction", multialignment="center")
    #     plt.annotate(r"With evolution from Sun et al.$^{c}$",
    #                  (0.68, -0.25),
    #                  xycoords="axes fraction", fontsize=7,
    #                  annotation_clip=False)
    #     plt.annotate("a: 2015ApJ...809...98A (IceCube Collab.)\n"
    #                  "b: 2018ApJ...852...72V (van Velzen) \n"
    #                  "c: 2015ApJ...812...33S (Sun et al.)",
    #                  (-0.15, -0.25),
    #                  xycoords="axes fraction", fontsize=7,
    #                  annotation_clip=False)
    #     plt.savefig(base_dir + "diffuse_flux_global_fit.pdf")
    #     plt.close()



    # calculate_transient(e_pdf_dict_template, biehl_jetted_rate,
    #                     "jetted TDEs (biehl)", zmax=2.5,
    #                     # diffuse_fit="Northern Tracks"
    #                     )



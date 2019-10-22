import matplotlib.pyplot as plt
import numpy as np
from flarestack.shared import plot_output_dir
from flarestack.utils.neutrino_cosmology import get_diffuse_flux_at_1GeV
from flarestack.misc.convert_diffuse_flux_contour import contour_95, \
    upper_contour, lower_contour, global_fit_e_range
from flarestack.analyses.tde.tde_cosmology import jetted_rate_uncertainties,\
    nonjetted_rate_uncertainties

# values taken from tde_cosmology.py

# Cat, Mass limit * M/10^6, Standard candle limit (GeV^-1 cm^-2 s^-1 sr^-1)

res = [
    # Fix me...
    # (1.83861002507e-07, 1.83861002507e-06, nonjetted_rate_uncertainties),
    (9.19158363155e-09, 9.26614817e-08, jetted_rate_uncertainties),

]

# TDE weights are defined relative to a BH mass of 10^6 solar masses.
# Assuming the standard dn/dM ~ M^-3/2, between 10^5.5 and 10^7.5, we end up
# with a mean BH mass of 10^6.5, corresponding to a mean weight of 10^0.5.

mean_tde_weight = 10**0.5

base_dir = plot_output_dir("analyses/tde/mass_weighted/")

plt.figure()
plt.subplot(111)

mass_range = np.logspace(5.5, 7.5, 5)

# # Plot 95% contour
#
# plt.fill_between(
#     upper_contour(global_fit_e_range, contour_95),
#     lower_contour(global_fit_e_range, contour_95),
#     color="k", label="IceCube\n" + r"diffuse flux$^{a}$",
#     alpha=.5,
# )

diffuse_norm, diffuse_gamma = get_diffuse_flux_at_1GeV()

plt.axhline(1.0, color="k",
            label="IceCube\n" + r"diffuse flux$^{a}$")

labels = [
    # "Non-jetted TDEs\n" +
    # r"($8^{+4}_{-4} \times 10^{-7}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
    "Jetted TDEs\n" +
    r"$(3^{+4}_{-2} \times 10^{-11}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
]

for k, (mass_lim, sc_lim, rate_uncertainties) in enumerate(res):

    [ul, ll] = rate_uncertainties[0]

    mass_lim /= (10**6 * diffuse_norm.value)
    sc_lim /= diffuse_norm.value

    color = ["orange", "blue"][k]

    sc_range = np.array([sc_lim for _ in mass_range])
    plt.errorbar(
        mass_range, sc_range,
        yerr=.25 * np.array([x for x in sc_range]),
        uplims=True, linestyle=":", color=color)
    plt.fill_between(mass_range, ul*sc_lim,
                     ll * sc_lim, alpha=0.05, color=color)

    # plt.plot(mass_range, mass_range*mass_lim, label=labels[k], color=color)
    plt.errorbar(
        mass_range, mass_range*mass_lim,
        yerr=.25 * np.array([x for x in mass_range*mass_lim]),
        uplims=True, label=labels[k], color=color)
    plt.fill_between(mass_range, mass_range*mass_lim*ul,
                     mass_range * mass_lim * ll, alpha=0.25, color=color)

# plt.plot([], [], color="k", linestyle="-",
#          label=r"$L_{\nu} \propto M_{BH}$")

plt.yscale("log")
plt.xscale("log")
l = plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, ncol=3)
for t in l.texts:
    t.set_multialignment('center')

plt.gca().add_artist(l)

lines = [plt.Line2D([], [], color="orange", linestyle=[":", "-"][k]) for k in
         range(2)]

plt.legend(
    lines, ["Standard Candle Limit", r"$L_{\nu} \propto M_{BH}$ Limit"],
    loc='upper center', bbox_to_anchor=(0.5, +1.13), fancybox=True, ncol=2
)

plt.ylabel(r"Diffuse flux fraction")
plt.xlabel(r"Mean TDE BH Mass ($\overline{M_{BH}}$) [$M_{\odot}$]")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.ylim(top=2)
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.annotate("IceCube \n Preliminary ", (0.7, 0.05), alpha=0.5, fontsize=15,
             xycoords="axes fraction", multialignment="center")
plt.annotate(r"With evolution from Sun et al.$^{b}$",
             (0.71, -0.3),
             xycoords="axes fraction", fontsize=7,
             annotation_clip=False)
plt.annotate("a: 2015ApJ...809...98A (IceCube Collab.)\n"
             "b: 2015ApJ...812...33S (Sun et al.)",
             (-0.13, -0.3),
             xycoords="axes fraction", fontsize=7,
             annotation_clip=False)
plt.savefig(base_dir + "diffuse_flux_global_fit_mass.pdf")
plt.close()

mass_range = np.logspace(5.5, 7.5, 5)

plt.figure()
plt.subplot(111)

e_range = np.logspace(2.73, 5.64, 3)

def f(energy, norm):
    return norm * energy ** -0.5

# Plot 95% contour

plt.fill_between(
    global_fit_e_range,
    global_fit_e_range ** 2 * upper_contour(global_fit_e_range, contour_95),
    global_fit_e_range ** 2 * lower_contour(global_fit_e_range, contour_95),
    color="k", label="IceCube\n" + r"diffuse flux$^{a}$",
    alpha=.5,
)

plt.plot(global_fit_e_range, f(global_fit_e_range, diffuse_norm),
         color="k")

labels = [
    # "Non-jetted TDEs\n" +
    # r"($8^{+4}_{-4} \times 10^{-7}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
    "Jetted TDEs\n" +
    r"$(3^{+4}_{-2} \times 10^{-11}$ Mpc$^{-3}$ yr$^{-1}$)$^{b}$",
]

for k, (mass_lim, sc_lim, rate_uncertainties) in enumerate(res):

    [ul, ll] = rate_uncertainties[0]

    mass_lim *= mean_tde_weight

    color = ["orange", "blue"][k]

    plt.errorbar(
        e_range, f(e_range, sc_lim),
        yerr=.25 * np.array([x for x in f(e_range, sc_lim)]),
        uplims=True, linestyle=":", color=color)

    sc_range = np.array([sc_lim for _ in mass_range])
    plt.fill_between(e_range, ul*f(e_range, sc_lim),
                     ll * f(e_range, sc_lim), alpha=0.05, color=color)

    plt.errorbar(
        e_range, f(e_range, mass_lim),
        yerr = .25 * np.array([x for x in f(e_range, mass_lim)]),
        uplims = True,  color = color, label=labels[k]
    )

    plt.fill_between(e_range, ul * f(e_range, mass_lim),
                     ll * f(e_range, mass_lim), alpha=0.25, color=color)

# plt.plot([], [], color="k", linestyle="-",
#          label=r"$L_{\nu} \propto M_{BH}$")

plt.yscale("log")
plt.xscale("log")
l = plt.legend(loc='upper center', bbox_to_anchor=(0.5, +1.35),
               fancybox=True, ncol=3)
for t in l.texts:
    t.set_multialignment('center')

plt.gca().add_artist(l)

lines = [plt.Line2D([], [], color="orange", linestyle=[":", "-"][k]) for k in
         range(2)]

linestlye_labels = [
    "Standard Candle Limit",
    r"$L_{\nu} \propto M_{BH}$ Limit ($\overline{M_{BH}} = 10^{6.5} M_{"
    r"\odot}$)"
]

plt.legend(
    lines, linestlye_labels,
    loc='upper center', bbox_to_anchor=(0.5, +1.17), fancybox=True, ncol=2
)

plt.ylabel(r"$E^{2}\frac{dN}{dE}$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
plt.xlabel(r"$E_{\nu}$ [GeV]")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.annotate("IceCube \n Preliminary ", (0.05, 0.05), alpha=0.5, fontsize=15,
             xycoords="axes fraction", multialignment="center")
plt.annotate(r"With evolution from Sun et al.$^{b}$",
             (0.68, -0.25),
             xycoords="axes fraction", fontsize=7,
             annotation_clip=False)
plt.annotate("a: 2015ApJ...809...98A (IceCube Collab.)\n"
             "b: 2015ApJ...812...33S (Sun et al.)",
             (-0.1, -0.25),
             xycoords="axes fraction", fontsize=7,
             annotation_clip=False)
plt.savefig(base_dir + "diffuse_flux_global_fit_energy.pdf")
plt.close()


import numpy as np
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.shared import plot_output_dir, flux_to_k, make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.tde.shared_TDE import tde_catalogues,\
    tde_weighted_catalogue_name
from flarestack.utils.neutrino_cosmology import get_diffuse_flux_at_1GeV
from flarestack.cluster import run_desy_cluster as rd
import math
import matplotlib.pyplot as plt
from flarestack.utils.custom_seasons import custom_dataset

analyses = dict()

# Initialise Injectors/LLHs

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

llh_time = {
    "Name": "FixedEndBox"
}

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Fit Weights?": True
}

gammas = [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
# gammas = [1.8, 2.0, 2.1, ]


# power_law_start_energy = [100]

cutoff_dict = dict()

injection_length = 100

raw = "analyses/tde/compare_mass_weighting/"

cat_res = dict()

for cat in ["jetted"]:

    name = raw + cat + "/"

    cat_path = tde_weighted_catalogue_name(cat)
    catalogue = np.load(cat_path)

    res = dict()

    closest_src = np.sort(catalogue, order="Distance (Mpc)")[0]

    for gamma in gammas:

        full_name = name + str(gamma) + "/"

        injection_time = llh_time = {
            "Name": "FixedEndBox",
        }

        injection_energy = dict(llh_energy)
        injection_energy["Gamma"] = gamma

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
            "Poisson Smear?": True,
        }

        scale = flux_to_k(reference_sensitivity(
            np.sin(closest_src["dec"]), gamma=gamma
        ) * 40 * math.sqrt(float(len(catalogue)))) / np.mean(
            catalogue["Relative Injection Weight"])

        mh_dict = {
            "name": full_name,
            "datasets": custom_dataset(txs_sample_v1, catalogue,
                                       llh_kwargs["LLH Time PDF"]),
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 5,
            "n_steps": 15
        }

        pkl_file = make_analysis_pickle(mh_dict)

        rd.submit_to_cluster(pkl_file, n_jobs=1000)

        # mh = MinimisationHandler(mh_dict)
        # mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"],
        #                n_trials=50)
        # mh.clear()

        res[gamma] = mh_dict

    cat_res[cat] = res

rd.wait_for_cluster()


for b, (cat_name, src_res) in enumerate(cat_res.iteritems()):

    name = raw + cat_name + "/"

    sens_livetime = []
    fracs = []
    disc_pots_livetime = []
    sens_e = []
    disc_e = []

    for (gamma, rh_dict) in sorted(res.iteritems()):
        try:
            rh = ResultsHandler(rh_dict["name"],
                                rh_dict["llh kwargs"],
                                rh_dict["catalogue"],
                                show_inj=True
                                )

            inj_time = injection_length * 60 * 60 * 24

            astro_sens, astro_disc = rh.astro_values(
                rh_dict["inj kwargs"]["Injection Energy PDF"])

            key = "Total Fluence (GeV cm^{-2} s^{-1})"

            e_key = "Mean Luminosity (erg/s)"

            sens_livetime.append(astro_sens[key] * inj_time)
            disc_pots_livetime.append(astro_disc[key] * inj_time)

            sens_e.append(astro_sens[e_key] * inj_time)
            disc_e.append(astro_disc[e_key] * inj_time)

            fracs.append(gamma)

        except OSError:
            pass

        # plt.plot(fracs, disc_pots, linestyle="--", color=cols[i])

    print fracs

    for j, [fluence, energy] in enumerate([[sens_livetime, sens_e],
                                          [disc_pots_livetime, disc_e]]):

        plt.figure()
        ax1 = plt.subplot(111)

        ax2 = ax1.twinx()

        cols = ["#00A6EB", "#F79646", "g", "r"]
        linestyle = ["-", "-"][j]

        ax1.plot(fracs, fluence, linestyle=linestyle)
        ax2.plot(fracs, energy, linestyle=linestyle)

        y_label = [r"Total Fluence [GeV cm$^{-2}$]",
                   r"Mean Isotropic-Equivalent $E_{\nu}$ (erg)"]

        ax2.grid(True, which='both')
        ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$]", fontsize=12)
        ax2.set_ylabel(r"Isotropic-Equivalent "
                       r"$E_{\nu} \times \frac{M_{BH}}{10^{6} M_{\odot}}$ ("
                       r"erg)")
        ax1.set_xlabel(r"Spectral Index ($\gamma$)")
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        for k, ax in enumerate([ax1, ax2]):
            y = [fluence, energy][k]

            try:

                ax.set_ylim(0.95 * min(y),
                            1.1 * max(y))
            except ValueError:
                pass

        plt.title("Stacked " + ["Sensitivity", "Discovery Potential"][j] +
                  " for " + cat_name + " TDEs")

        ax1.legend(loc='upper left', fancybox=True, framealpha=0.)
        plt.tight_layout()
        plt.savefig(plot_output_dir(name) +
                    ["sens", "disc"][j] + "_" + cat_name + ".pdf")
        plt.close()
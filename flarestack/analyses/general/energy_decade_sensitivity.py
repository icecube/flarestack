import numpy as np
import os
import cPickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
from flarestack.shared import plot_output_dir, flux_to_k, analysis_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.reference_sensitivity import reference_sensitivity
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt

# Initialise Injectors/LLHs

llh_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
}

injection_time = {
    "Name": "Steady",
}

llh_time = {
    "Name": "Steady",
}

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
}

base_name = "benchmarks/energy_decades"

# sindecs = np.linspace(0.90, -0.90, 13)
# sindecs = np.linspace(0.75, -0.75, 7)
sindecs = np.linspace(0.5, -0.5, 3)
# sindecs = [-0.5]

energies = np.logspace(2, 7, 11)
bins = zip(energies[:-1], energies[1:])
# print bins
# raw_input("prompt")

analyses = dict()

for sindec in sindecs:

    source_res = dict()

    cat_path = ps_catalogue_name(sindec)

    subname = base_name + "/sindec=" + '{0:.2f}'.format(sindec) + "/"

    southern = sindec < np.sin(np.radians(-5))

    for i, (e_min, e_max) in enumerate(bins):

        name = subname + '{0:.2f}'.format(e_min) +"/"

        injection_energy = dict(llh_energy)
        injection_energy["E Min"] = e_min
        injection_energy["E Max"] = e_max

        inj_kwargs = {
            "Injection Energy PDF": injection_energy,
            "Injection Time PDF": injection_time,
            "Poisson Smear?": True,
        }

        e_center = 0.5 * (np.log10(e_min) + np.log10(e_max))

        parabola_min = 5 + 0.5 * (1 + np.sign(-sindec -np.sin(np.radians(5))))

        e_scale = np.exp(0.5*(e_center - parabola_min) ** 2)

        scale = flux_to_k(reference_sensitivity(sindec)) * 100 * e_scale

        mh_dict = {
            "name": name,
            "datasets": ps_7year[-2:-1],
            "catalogue": cat_path,
            "inj kwargs": inj_kwargs,
            "llh kwargs": llh_kwargs,
            "scale": scale,
            "n_trials": 10,
            "n_steps": 20
        }

        analysis_path = analysis_dir + name

        try:
            os.makedirs(analysis_path)
        except OSError:
            pass

        pkl_file = analysis_path + "dict.pkl"

        with open(pkl_file, "wb") as f:
            Pickle.dump(mh_dict, f)

        if not (southern and e_min < 1e4):
            rd.submit_to_cluster(pkl_file, n_jobs=100)

            # mh = MinimisationHandler(mh_dict)
            # mh.iterate_run(mh_dict["scale"], n_steps=20,
            #                n_trials=mh_dict["n_trials"])
            # mh.clear()

        source_res[i] = mh_dict

    analyses[sindec] = source_res

rd.wait_for_cluster()

plt.figure()

for (sindec, source_res) in analyses.iteritems():

    x = []

    sens = []
    disc_pots = []

    for i, (e_min, rh_dict) in enumerate(source_res.iteritems()):
        try:
            rh = ResultsHandler(rh_dict["name"], rh_dict["llh kwargs"],
                                rh_dict["catalogue"])
            sens += [rh.sensitivity for _ in range(2)]
            x += [z for z in bins[i]]
            disc_pots.append(rh.disc_potential)
        except OSError:
            pass

    label = r"$\sin(\delta)$ = " + str(sindec)
    plt.plot(x, sens, label=label)

plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"$E^2 \mathrm{d}N /\mathrm{d}E$ [ GeV cm$^{-2}$ s$^{-1}$ ]")
plt.xlabel(r"$E_{\nu}$ [GeV]")
path = plot_output_dir(base_name) + "/comparison.pdf"
plt.tight_layout()
plt.savefig(path)

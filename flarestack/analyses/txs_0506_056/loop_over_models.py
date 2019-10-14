from __future__ import print_function
from builtins import str
from builtins import range
import numpy as np
import os
import pickle as Pickle
from flarestack.core.results import ResultsHandler
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict,\
    IC86_234_dict
from flarestack.shared import flux_to_k, make_analysis_pickle, plot_output_dir
from flarestack.core.minimisation import MinimisationHandler
from flarestack.cluster import run_desy_cluster as rd
import matplotlib.pyplot as plt
from flarestack.analyses.txs_0506_056.make_txs_catalogue import \
    txs_catalogue, txs_cat_path
from flarestack.analyses.txs_0506_056.load_gao_spectral_models import \
    spline_name
from flarestack.utils.custom_dataset import custom_dataset

base_dir = "analyses/txs_0506_056/loop_gao_models/"

res_dict = dict()

n_range = list(range(160))

for i in n_range:

    # Initialise Injectors/LLHs

    # Set up what is "injected" into the fake dataset. This is a simulated source

    # Use a source that is constant in time

    injection_time = {
        "Name": "FixedEndBox"
    }

    # Use a source with a spectral index of -2, with an energy range between
    # 100 GeV and 10 Pev (10**7 GeV).

    injection_energy = {
        "Name": "Spline",
        "Spline Path": spline_name(i)
    }

    # Fix injection time/energy PDFs, and use "Poisson Smearing" to simulate
    # random variations in detected neutrino numbers

    inj_kwargs = {
        "Injection Energy PDF": injection_energy,
        "Injection Time PDF": injection_time,
        "Poisson Smear?": True,
    }

    # Set up the "likelihood" arguments, which determine how the fake data is
    # analysed.

    # Look for a source that is constant in time

    llh_time = injection_time

    # Try to fit a power law to the data

    llh_energy = {
        "Name": "Power Law"
    }

    # Set up a likelihood that fits the number of signal events (n_s), and also
    # the spectral index (gamma) of the source

    llh_kwargs = {
        "name": "standard",
        "LLH Energy PDF": llh_energy,
        "LLH Time PDF": llh_time,
    }

    # Takes a guess at the correct flux scale, based on previous IceCube results

    scale = flux_to_k(2.) * 10**((160.-i)/160.)

    # Assign a unique name for each different minimisation handler dictionary

    name = base_dir + str(i) + "/"

    # Creates the Minimisation Handler dictionary, which contains all relevant
    # information to run an analysis

    mh_dict = {
        "name": name,
        "mh_name": "fixed_weights",
        "datasets": [IC86_234_dict],
        "catalogue": txs_cat_path,
        "inj kwargs": inj_kwargs,
        "llh_dict": llh_kwargs,
        "scale": scale,
        "n_trials": 100,
        "n_steps": 11
    }

    pkl_file = make_analysis_pickle(mh_dict)

    # Creates a Minimisation Handler using the dictionary, and runs the trials

    # mh_power_law = MinimisationHandler.create(mh_dict_power_law)
    # mh_power_law.iterate_run(mh_dict_power_law["scale"], n_steps=mh_dict_power_law["n_steps"],
    #                n_trials=mh_dict_power_law["n_trials"])

    rd.submit_to_cluster(pkl_file, n_jobs=5)

    res_dict[i] = mh_dict


rd.wait_for_cluster()

sens = []
disc = []

for i in n_range:
    mh_dict = res_dict[i]

    # Creates a Results Handler to analyse the results, and calculate the
    # sensitivity. This is the flux that needs to arrive at Earth, in order for
    # IceCube to see an overfluctuation 90% of the time. Prints this information.

    rh = ResultsHandler(mh_dict)
    sens.append(rh.sensitivity)
    disc.append(rh.disc_potential)


for i, vals in enumerate([sens, disc]):

    label = ["Sensitivity", "Discovery Potential"][i]

    plt.figure()
    plt.plot(n_range, vals)
    plt.ylabel("Ratio to Model")
    plt.xlabel("Model Number")
    plt.title(label + " during TXS Neutrino Flare")

    savepath = plot_output_dir(base_dir) + label + ".pdf"

    try:
        os.makedirs(os.path.dirname(savepath))
    except OSError:
        pass

    print("Saving to", savepath)
    plt.savefig(savepath)
    plt.close()





from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from flarestack.utils.simulate_catalogue import simulate_transients
from flarestack.analyses.ccsn import get_sn_type_rate
from flarestack.shared import plot_output_dir

sn_types = ["IIn"]

for sn in sn_types:
    rate = get_sn_type_rate(sn_type=sn)

    cat = simulate_transients(1, rate, local_z=0.2)

    name = "analyses/gen2_wp/cat_" + sn + "/"

    savedir = plot_output_dir(name)

    try:
        os.makedirs(savedir)
    except OSError:
        pass

    plt.figure()
    plt.hist(cat)
    plt.yscale("log")
    plt.savefig(savedir + "hist.pdf")
    plt.close()

    catpath = savedir + "cat.npy"
    print("Saving to", catpath)
    np.save(catpath, cat)

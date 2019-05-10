from __future__ import print_function
import numpy as np
from flarestack.shared import catalogue_dir, plot_output_dir
import matplotlib.pyplot as plt

cats = [
    "jetted",
    "gold",
    "obscured",
    "silver"
]

for j, cat in enumerate(cats):

    cat_path = catalogue_dir + "TDEs/TDE_" + cat + "_catalogue.npy"
    catalogue = np.load(cat_path)

    times = catalogue["End Time (MJD)"] - catalogue["Start Time (MJD)"]

    print(cat, "Livetime", np.sum(times))

    savepath = plot_output_dir("analyses/tde/") + cat + "_hist.pdf"
    plt.figure()
    plt.hist(times, range=(0, max(times)), bins=20)
    plt.xlabel("Search Window (Days)")
    plt.title(cat + " TDEs")
    plt.savefig(savepath)
    plt.close()





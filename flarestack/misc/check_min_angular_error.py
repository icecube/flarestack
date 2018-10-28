import numpy as np
import os
import matplotlib.pyplot as plt
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.utils.dataset_loader import data_loader
from flarestack.shared import plots_dir, min_angular_err

ang_err_dir = plots_dir + "angular_error/"

for season in txs_sample_v1:
    sample_dir = ang_err_dir + season["Data Sample"] + "/"

    try:
        os.makedirs(sample_dir)
    except OSError:
        pass

    if isinstance(season["exp_path"], list):
        exp = np.concatenate(
            tuple([np.load(x) for x in season["exp_path"]]))
    else:
        exp = np.load(season["exp_path"])

    plt.figure()
    plt.hist(np.degrees(exp["sigma"]), bins=100)
    plt.yscale("log")
    plt.savefig(sample_dir + season["Name"] + "_RAW.pdf")
    plt.close()

    mask = exp["sigma"] < min_angular_err

    print

    print season["Name"], "(reading raw data)"

    print np.sum(mask), "events have an angular error less than",
    print np.degrees(min_angular_err), "degrees"
    print "This is out of", len(mask), "total events"

    exp = data_loader(season["exp_path"])

    mask = exp["sigma"] < min_angular_err

    plt.figure()
    plt.hist(np.degrees(exp["sigma"]), bins=100)
    plt.yscale("log")
    plt.savefig(sample_dir + season["Name"] + "_corrected.pdf")
    plt.close()

    print

    print season["Name"], "(read in with data_loader)"

    print np.sum(mask), "events have an angular error less than",
    print np.degrees(min_angular_err), "degrees"
    print "This is out of", len(mask), "total events"

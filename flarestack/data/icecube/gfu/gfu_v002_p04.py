"""File containing links to data samples used (GFU).

This was the complete GFU data samle, extending from 2011-2018, that was used
in Thomas Kintscher's final unblinding of GFU for the application of the
'Online Flare Search' method on archival data.
"""
import flarestack.shared
import numpy as np

gfu_data_dir = flarestack.shared.dataset_dir + "gfu/version-002-p04/"
grl_dir = gfu_data_dir + "GRL/"

gfu_dict = {
    "Data Sample": "gfu_v002_p04",
    "sinDec bins": np.unique(np.concatenate([
        np.linspace(-1., -0.9, 2 + 1),
        np.linspace(-0.9, -0.2, 8 + 1),
        np.linspace(-0.2, 0.2, 15 + 1),
        np.linspace(0.2, 0.9, 12 + 1),
        np.linspace(0.9, 1., 2 + 1),
    ])),
    "MJD Time Key": "time",
    "Name": "GFU_v002_p04",
    "exp_path": [
        gfu_data_dir + "IC86_2011_data.npy",
        gfu_data_dir + "IC86_2012_data.npy",
        gfu_data_dir + "IC86_2013_data.npy",
        gfu_data_dir + "IC86_2014_data.npy",
        gfu_data_dir + "IC86_2015_data.npy",
        gfu_data_dir + "IC86_2016_data.npy",
        gfu_data_dir + "IC86_2017_data.npy",
        gfu_data_dir + "IC86_2018_data.npy"
    ],
    "mc_path": gfu_data_dir + "IC86_2011_MC.npy",
    "grl_path": [
        grl_dir + "IC86_2011_data.npy",
        grl_dir + "IC86_2012_data.npy",
        grl_dir + "IC86_2013_data.npy",
        grl_dir + "IC86_2014_data.npy",
        grl_dir + "IC86_2015_data.npy",
        grl_dir + "IC86_2016_data.npy",
        grl_dir + "IC86_2017_data.npy",
        grl_dir + "IC86_2018_data.npy"
    ]
}

gfu_v002_p04 = [gfu_dict]

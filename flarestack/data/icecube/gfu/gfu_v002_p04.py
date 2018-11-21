"""File containing links to data samples used (GFU).

Path to local copy of point source tracks, downloaded on 16/05/18 from
/data/ana .. /current, with the following readme:

    * better pull-correction using splines, instead of polynomial fits

This is the sample that was used for the original TXS analysis in 2017
(NOT THE ONE THAT INCLUDED ADDITIONAL DATA UP TO SPRING 2018!)
"""
import flarestack.shared
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_7year
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

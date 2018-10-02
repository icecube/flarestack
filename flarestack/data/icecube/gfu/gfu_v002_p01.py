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

gfu_data_dir = flarestack.shared.dataset_dir + "gfu/version-002-p01/"

gfu_dict = {
    "Data Sample": "gfu_v002_p01",
    "sinDec bins": np.unique(np.concatenate([
        np.linspace(-1., -0.9, 2 + 1),
        np.linspace(-0.9, -0.2, 8 + 1),
        np.linspace(-0.2, 0.2, 15 + 1),
        np.linspace(0.2, 0.9, 12 + 1),
        np.linspace(0.9, 1., 2 + 1),
    ])),
    "MJD Time Key": "time",
    "Name": "GFU_v002_p01",
    "exp_path": [
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2015.npy",
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2016.npy",
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2017.npy"
    ],
    "mc_path": gfu_data_dir + "SplineMPEmax.MuEx.MC.npy",
    "grl_path": gfu_data_dir + "SplineMPEmax.MuEx.GRL.npy"
}

gfu_v002_p01 = [gfu_dict]

txs_sample_v1 = ps_7year + gfu_v002_p01
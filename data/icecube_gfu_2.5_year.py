from shared import dataset_dir
import numpy as np

data_dir = dataset_dir + "gfu_160518/"

gfu_dict = {
    "Data Sample": "gfu_2.5_year",
    "sinDec bins": np.unique(np.concatenate([
        np.linspace(-1., -0.9, 2 + 1),
        np.linspace(-0.9, -0.2, 8 + 1),
        np.linspace(-0.2, 0.2, 15 + 1),
        np.linspace(0.2, 0.9, 12 + 1),
        np.linspace(0.9, 1., 2 + 1),
    ])),
    "MJD Time Key": "time"
}

gfu_2015 = {
    "Name": "GFU_IC86_2015",
    "exp_path": data_dir + "SplineMPEmax.MuEx.IC86-2015.npy",
    "mc_path": data_dir + "SplineMPEmax.MuEx.MC.npy",
    "grl_path": data_dir + "SplineMPEmax.MuEx.GRL.npy"
}
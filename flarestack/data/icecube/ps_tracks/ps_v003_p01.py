"""PS Tracks v003_p01, as used by Tessa in the 10 year PS analysis.
"""
from flarestack.shared import dataset_dir
import numpy as np

ps_data_dir = dataset_dir + "ps_tracks/version-003-p01/"

ps_dict = {
    "Data Sample": "ps_tracks_v003_p01",
    "sinDec bins": np.unique(np.concatenate([
            np.linspace(-1., -0.9, 2 + 1),
            np.linspace(-0.9, -0.2, 8 + 1),
            np.linspace(-0.2, 0.2, 15 + 1),
            np.linspace(0.2, 0.9, 12 + 1),
            np.linspace(0.9, 1., 2 + 1),
        ])),
    "MJD Time Key": "time"
}

IC40_dict = {
    "Name": "IC40",
    "exp_path": ps_data_dir + "IC40_exp.npy",
    "mc_path": ps_data_dir + "IC40_MC.npy",
    "grl_path": ps_data_dir + "IC40_exp.npy"
}
IC40_dict.update(ps_dict)

IC59_dict = {
    "Name": "IC59",
    "exp_path": ps_data_dir + "IC59_exp.npy",
    "mc_path": ps_data_dir + "IC59_MC.npy",
    "grl_path": ps_data_dir + "IC59_exp.npy"
}
IC59_dict.update(ps_dict)


IC79_dict = {
    "Name": "IC79",
    "exp_path": ps_data_dir + "IC79_exp.npy",
    "mc_path": ps_data_dir + "IC79_MC.npy",
    "grl_path": ps_data_dir + "IC79_exp.npy"
}
IC79_dict.update(ps_dict)


IC86_1_dict = {
    "Name": "IC86_1",
    "exp_path": ps_data_dir + "IC86_2011_exp.npy",
    "mc_path": ps_data_dir + "IC86_2011_MC.npy",
    "grl_path": ps_data_dir + "IC86_2011_exp.npy"
}
IC86_1_dict.update(ps_dict)


IC86_234567_dict = {
    "Name": "IC86_234567",
    "exp_path": [
        ps_data_dir + "IC86_2012_exp.npy",
        ps_data_dir + "IC86_2013_exp.npy",
        ps_data_dir + "IC86_2014_exp.npy",
        ps_data_dir + "IC86_2015_exp.npy",
        ps_data_dir + "IC86_2016_exp.npy",
        ps_data_dir + "IC86_2017_exp.npy",
        ],
    "mc_path": ps_data_dir + "IC86_2012_MC.npy",
    "grl_path": [
        ps_data_dir + "IC86_2012_exp.npy",
        ps_data_dir + "IC86_2013_exp.npy",
        ps_data_dir + "IC86_2014_exp.npy",
        ps_data_dir + "IC86_2015_exp.npy",
        ps_data_dir + "IC86_2016_exp.npy",
        ps_data_dir + "IC86_2017_exp.npy"
    ]
}

IC86_234567_dict.update(ps_dict)

ps_10year = [
    IC40_dict, IC59_dict, IC79_dict, IC86_1_dict, IC86_234567_dict,
]
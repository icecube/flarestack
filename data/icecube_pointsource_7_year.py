from shared import dataset_dir
import numpy as np

ps_data_dir = dataset_dir + "ps_tracks_240418/"

ps_dict = {
    "Data Sample": "ps_7_year",
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
    "mc_path": ps_data_dir + "IC40_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC40_GRL.npy"
}
IC40_dict.update(ps_dict)

IC59_dict = {
    "Name": "IC59",
    "exp_path": ps_data_dir + "IC59_exp.npy",
    "mc_path": ps_data_dir + "IC59_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC59_GRL.npy"
}
IC59_dict.update(ps_dict)


IC79_dict = {
    "Name": "IC79",
    "exp_path": ps_data_dir + "IC79b_exp.npy",
    "mc_path": ps_data_dir + "IC79b_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC79b_GRL.npy"
}
IC79_dict.update(ps_dict)


IC86_1_dict = {
    "Name": "IC86_1",
    "exp_path": ps_data_dir + "IC86_exp.npy",
    "mc_path": ps_data_dir + "IC86_corrected_MC.npy",
    "grl_path": ps_data_dir + "IC86_GRL.npy"
}
IC86_1_dict.update(ps_dict)


IC86_234_dict = {
    "Name": "IC86_234",
    "exp_path": [
        ps_data_dir + "IC86-2012_exp_v2.npy",
        ps_data_dir + "IC86-2013_exp_v2.npy",
        ps_data_dir + "IC86-2014_exp_v2.npy"
        ],
    "mc_path": ps_data_dir + "IC86-2012_corrected_MC_v2.npy",
    "grl_path": [
        ps_data_dir + "IC86-2012_GRL.npy",
        ps_data_dir + "IC86-2013_GRL.npy",
        ps_data_dir + "IC86-2014_GRL.npy"
    ]
}
IC86_234_dict.update(ps_dict)

# IC86_2013_dict = {
#     "Name": "IC86_2013",
#     "exp_path": diffuse_data_dir + "IC86-2013_exp_v2.npy",
#     "mc_path": diffuse_data_dir + "IC86-2012_corrected_MC_v2.npy",
#     "grl_path": diffuse_data_dir + "IC86-2013_GRL.npy"
# }
# IC86_2013_dict.update(ps_dict)
#
# IC86_2014_dict = {
#     "Name": "IC86_2014",
#     "exp_path": diffuse_data_dir + "IC86-2014_exp_v2.npy",
#     "mc_path": diffuse_data_dir + "IC86-2012_corrected_MC_v2.npy",
#     "grl_path": diffuse_data_dir + "IC86-2014_GRL.npy"
# }
# IC86_2014_dict.update(ps_dict)

ps_7year = [
    IC40_dict, IC59_dict, IC79_dict, IC86_1_dict, IC86_234_dict,
]

# ps_7year = [IC79_dict, IC86_1_dict]

ps_7986 = [IC79_dict, IC86_1_dict]

from shared import dataset_dir
import numpy as np
import os
from data.icecube_pointsource_7_year import ps_data_dir
from data.icecube_gfu_2point5_year import gfu_data_dir

diffuse_data_dir = dataset_dir + "northern_tracks_140518/"

diffuse_grl_dir = dataset_dir + "northern_tracks_grl/"

try:
    os.makedirs(diffuse_grl_dir)
except OSError:
    pass


def diffuse_grl_pathname(season_dict):
    return diffuse_grl_dir + season_dict["Name"] + "_grl.npy"

diffuse_dict = {
    "Data Sample": "diffuse_8_year",
    "sinDec bins": np.unique(np.concatenate([
        np.linspace(-0.05, 0.2, 8 + 1),
        np.linspace(0.2, 0.9, 12 + 1),
        np.linspace(0.9, 1., 2 + 1),
    ])),
    "MJD Time Key": "st_time"
}

diffuse_IC59 = {
    "Name": "IC59",
    "exp_path": diffuse_data_dir + "dataset_8yr_fit_IC59_exp.npy",
    "mc_path": diffuse_data_dir + "dataset_8yr_fit_IC59_MC_compressed.npy",
    "grl_path": ps_data_dir + "IC59_GRL.npy"
}
diffuse_IC59.update(diffuse_dict)

diffuse_IC79 = {
    "Name": "IC79",
    "exp_path": diffuse_data_dir + "dataset_8yr_fit_IC79_exp.npy",
    "mc_path": diffuse_data_dir + "dataset_8yr_fit_IC79_MC_compressed.npy",
    "grl_path": ps_data_dir + "IC79b_GRL.npy"
}
diffuse_IC79.update(diffuse_dict)

diffuse_IC86_1 = {
    "Name": "IC86_1",
    "exp_path": diffuse_data_dir + "dataset_8yr_fit_IC86_2011_exp.npy",
    "mc_path": diffuse_data_dir + "dataset_8yr_fit_IC86_2011_MC_compressed.npy",
    "grl_path": ps_data_dir + "IC86_GRL.npy"
}
diffuse_IC86_1.update(diffuse_dict)

diffuse_IC86_23456 = {
    "Name": "IC86_23456",
    "exp_path": diffuse_data_dir + "dataset_8yr_fit_IC86_2012_16_exp.npy",
    "mc_path": diffuse_data_dir +
               "dataset_8yr_fit_IC86_2012_16_MC_compressed.npy",
    "grl_path": [
        ps_data_dir + "IC86-2012_GRL.npy",
        ps_data_dir + "IC86-2013_GRL.npy",
        ps_data_dir + "IC86-2014_GRL.npy",
        gfu_data_dir + "SplineMPEmax.MuEx.GRL.npy"
    ]
}
diffuse_IC86_23456.update(diffuse_dict)

diffuse_8year = [diffuse_IC59, diffuse_IC79, diffuse_IC86_1, diffuse_IC86_23456]

# exp_path = diffuse_data_dir + "dataset_8yr_fit_IC59_MC_compressed.npy"
#
# exp = np.load(exp_path)
#
# print exp.dtype.names
#
# print np.sin(max(exp["dec"])), np.sin(min(exp["dec"]))

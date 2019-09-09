"""File containing links to data samples used (GFU).

Path to local copy of point source tracks, downloaded on 16/05/18 from
/data/ana .. /current, with the following readme:

    * better pull-correction using splines, instead of polynomial fits

This is the sample that was used for the original TXS analysis in 2017
(NOT THE ONE THAT INCLUDED ADDITIONAL DATA UP TO SPRING 2018!)
"""
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.data.icecube.ic_season import IceCubeDataset, \
    IceCubeSeason, icecube_dataset_dir
from flarestack.data.icecube.gfu import gfu_binning
import numpy as np

gfu_data_dir = icecube_dataset_dir + "gfu/version-002-p01/"

gfu_v002_p01 = IceCubeDataset()

sample_name = "gfu_v002_p01"

gfu_season = IceCubeSeason(
    season_name=sample_name,
    sample_name=sample_name,
    exp_path=[
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2015.npy",
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2016.npy",
        gfu_data_dir + "SplineMPEmax.MuEx.IC86-2017.npy"
    ],
    mc_path=gfu_data_dir + "SplineMPEmax.MuEx.MC.npy",
    grl_path=gfu_data_dir + "SplineMPEmax.MuEx.GRL.npy",
    sin_dec_bins=gfu_binning[0],
    log_e_bins=gfu_binning[1]
)

gfu_v002_p01.add_season(gfu_season)

txs_sample_v1 = IceCubeDataset()

for season in ps_v002_p01.values():
    txs_sample_v1.add_season(season)

txs_sample_v1.add_season(gfu_season)
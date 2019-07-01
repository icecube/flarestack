"""File containing links to data samples used (GFU).

This was the complete GFU data samle, extending from 2011-2018, that was used
in Thomas Kintscher's final unblinding of GFU for the application of the
'Online Flare Search' method on archival data.
"""
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import ps_v002_p01
from flarestack.data.icecube.ic_season import IceCubeDataset, \
    IceCubeSeason, icecube_dataset_dir
from flarestack.data.icecube.gfu import gfu_binning
import numpy as np

gfu_data_dir = icecube_dataset_dir + "gfu/version-002-p04/"
grl_dir = gfu_data_dir + "GRL/"


def get_exp_path(year):
    return gfu_data_dir + "IC86_{0}_data.npy".format(year)


def get_grl_path(year):
    return grl_dir + "IC86_{0}_data.npy".format(year)


all_years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]

gfu_v002_p04 = IceCubeDataset()

sample_name = "gfu_v002_p04"

gfu_8_year = IceCubeSeason(
    season_name="gfu_8_year",
    sample_name=sample_name,
    exp_path=[get_exp_path(x) for x in all_years],
    mc_path=gfu_data_dir + "IC86_2011_MC.npy",
    grl_path=[get_grl_path(x) for x in all_years],
    sin_dec_bins=gfu_binning[0],
    log_e_bins=gfu_binning[1]
)

gfu_v002_p04.add_season(gfu_8_year)

for year in all_years:
    gfu_year = IceCubeSeason(
        season_name="IC86_{0}".format(year),
        sample_name=sample_name,
        exp_path=get_exp_path(year),
        mc_path=gfu_data_dir + "IC86_2011_MC.npy",
        grl_path=get_grl_path(year),
        sin_dec_bins=gfu_binning[0],
        log_e_bins=gfu_binning[1]
    )
    gfu_v002_p04.add_subseason(gfu_year)

"""PS Tracks v003_p02, as used by Alex Pizzuto in the Anita analysis.
It includes runs 125865-125867 with 2 dropped strings.
"""
from flarestack.data.icecube.ic_season import IceCubeSeason, \
    IceCubeDataset, icecube_dataset_dir
from flarestack.data.icecube.ps_tracks import get_ps_binning
import numpy as np

ps_data_dir = icecube_dataset_dir + "ps_tracks/version-003-p02/"
grl_data_dir = ps_data_dir + "GRL/"

ps_v003_p02 = IceCubeDataset()

sample_name = "ps_tracks_v003_p02"


# Add in old seasons before full detector operation, and IC86_1


def old_ic_season(season):
    return IceCubeSeason(
        season_name=season,
        sample_name=sample_name,
        exp_path=ps_data_dir + "{0}_exp.npy".format(season),
        mc_path=ps_data_dir + "{0}_MC.npy".format(season),
        grl_path=grl_data_dir + "{0}_exp.npy".format(season),
        sin_dec_bins=get_ps_binning(season)[0],
        log_e_bins=get_ps_binning(season)[1]
    )


old_seasons = ["IC40", "IC59", "IC79", "IC86_2011"]

for season in old_seasons:
    ps_v003_p02.add_season(old_ic_season(season))

# Add in combined IC86 2012-2017 seasons

new_years = ["2012", "2013", "2014", "2015", "2016", "2017"]

ic86_234567 = IceCubeSeason(
    season_name="IC86_2012_17",
    sample_name=sample_name,
    exp_path=[
        ps_data_dir + "IC86_{0}_exp.npy".format(x) for x in new_years
    ],
    mc_path=ps_data_dir + "IC86_2012_MC.npy",
    grl_path=[
        grl_data_dir + "IC86_{0}_exp.npy".format(x) for x in new_years
    ],
    sin_dec_bins=get_ps_binning("IC86_2012")[0],
    log_e_bins=get_ps_binning("IC86_2012")[1]
)

ps_v003_p02.add_season(ic86_234567)


# Add in each new season as an optional subseason

def ic86_new_season(year):
    return IceCubeSeason(
        season_name="IC86_{0}".format(year),
        sample_name=sample_name,
        exp_path=ps_data_dir + "IC86_{0}_exp.npy".format(year),
        mc_path=ps_data_dir + "IC86_2012_MC.npy",
        grl_path=grl_data_dir + "IC86_{0}_exp.npy".format(year),
        sin_dec_bins=get_ps_binning(f"IC86_{year}")[0],
        log_e_bins=get_ps_binning(f"IC86_{year}")[1]
    )


for year in new_years:
    ps_v003_p02.add_subseason(ic86_new_season(year))

"""File containing links to data samples used (pointsource tracks).

Path to local copy of point source tracks, downloaded on 24/04/18 from
/data/ana .. /current, with the following readme:

    This directory contains a patched version of Stefan Coenders' original npy
    files prepared for the 7yr time integrated paper (version-002p00).
    IC40 through IC86 2011 are exactly the same as used in the paper.

    The files contain some track events which overlap with the MESE sample,
    but not all MESE events are present. Do not use these file if you want
    to perform MESE + PS style analysis.

    IC86 2012-2014 has one small patch to fix a known bug in the original files:

    FIX - updated per event angular uncertainties for IC86 2012-2014 files

    This fix is done by

    (1) applying median angular resolution from bootstrap method in cases
    where the sigma paraboloid fit fails
    (2) apply the pull correction splines from Asen's time dependent
    unblinding as they are currently our best pull correction of
    the 7yr sample. See: https://docushare.icecube.wisc.edu/dsweb/Get/Document-
    77805/christov_PSCall_27.6.2016.pdf

    Below is a description of all the years and their corresponding files.

    Josh - Nov 6, 2017

    IC40:
      Data File  IC40_exp.npy
      MC File    IC40_corrected_MC.npy

    IC59:
      Data File  IC59_exp.npy
      MC File    IC59_corrected_MC.npy

    IC79:
      Data File  IC79b_exp.npy
      MC File    IC79b_corrected_MC.npy

    IC86, 2011:
      Data File  IC86_exp.npy
      MC File    IC86_corrected_MC.npy

    IC86, 2012:
      Data File  IC86-2012_exp_v2.npy
      MC File    IC86-2012_corrected_MC_v2.npy

    IC86, 2013:
      Data File  IC86-2013_exp_v2.npy
      MC File    IC86-2013_corrected_MC_v2.npy

    IC86, 2014:
      Data File  IC86-2014_exp_v2.npy
      MC File    IC86-2014_corrected_MC_v2.npy

"""
from flarestack.data.icecube.ic_season import IceCubeDataset, \
    IceCubeSeason, icecube_dataset_dir
from flarestack.data.icecube.ps_tracks import get_ps_binning
import numpy as np
import copy

ps_data_dir = icecube_dataset_dir + "ps_tracks/version-002-p01/"

ps_v002_p01 = IceCubeDataset()

sample_name = "ps_tracks_v002_p01"

ic40 = IceCubeSeason(
    season_name="IC40",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC40_exp.npy",
    mc_path=ps_data_dir + "IC40_corrected_MC.npy",
    grl_path=ps_data_dir + "IC40_GRL.npy",
    sin_dec_bins=get_ps_binning("IC40")[0],
    log_e_bins=get_ps_binning("IC40")[1]
)

ps_v002_p01.add_season(ic40)


ic59 = IceCubeSeason(
    season_name="IC59",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC59_exp.npy",
    mc_path=ps_data_dir + "IC59_corrected_MC.npy",
    grl_path=ps_data_dir + "IC59_GRL.npy",
    sin_dec_bins=get_ps_binning("IC59")[0],
    log_e_bins=get_ps_binning("IC59")[1]
)

ps_v002_p01.add_season(ic59)

ic79 = IceCubeSeason(
    season_name="IC79",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC79b_exp.npy",
    mc_path=ps_data_dir + "IC79b_corrected_MC.npy",
    grl_path=ps_data_dir + "IC79b_GRL.npy",
    sin_dec_bins=get_ps_binning("IC79")[0],
    log_e_bins=get_ps_binning("IC79")[1]
)
ps_v002_p01.add_season(ic79)

boundary = np.sin(np.radians(-5.))  # North/South transition boundary

ic86_1 = IceCubeSeason(
    season_name="IC86_1",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC86_exp.npy",
    mc_path=ps_data_dir + "IC86_corrected_MC.npy",
    grl_path=ps_data_dir + "IC86_GRL.npy",
    sin_dec_bins=get_ps_binning("IC86_2011")[0],
    log_e_bins=get_ps_binning("IC86_2011")[1]
)

ps_v002_p01.add_season(ic86_1)

# Add optional subseasons for IC86 2, 3, and 4, that can be called instead of
# the combined season

for i in range(2, 5):
    ic86_i = IceCubeSeason(
        season_name="IC86_{0}".format(i),
        sample_name=sample_name,
        exp_path=ps_data_dir + "IC86-201{0}_exp_v2.npy".format(i),
        mc_path=ps_data_dir + "IC86-2012_corrected_MC_v2.npy",
        grl_path=ps_data_dir + "IC86-201{0}_GRL.npy".format(i),
        sin_dec_bins=get_ps_binning("IC86_2012")[0],
        log_e_bins=get_ps_binning("IC86_2012")[1]
    )
    ps_v002_p01.add_subseason(ic86_i)

ic86_234 = IceCubeSeason(
    season_name="IC86_234",
    sample_name=sample_name,
    exp_path=[
        ps_data_dir + "IC86-2012_exp_v2.npy",
        ps_data_dir + "IC86-2013_exp_v2.npy",
        ps_data_dir + "IC86-2014_exp_v2.npy"
    ],
    mc_path=ps_data_dir + "IC86-2012_corrected_MC_v2.npy",
    grl_path=[
        ps_data_dir + "IC86-2012_GRL.npy",
        ps_data_dir + "IC86-2013_GRL.npy",
        ps_data_dir + "IC86-2014_GRL.npy"
    ],
    sin_dec_bins=get_ps_binning("IC86_2012")[0],
    log_e_bins=get_ps_binning("IC86_2012")[1]
)

ps_v002_p01.add_season(ic86_234)

ps_3_systematic_set = IceCubeDataset()

for i, season_name in enumerate(["IC79", "IC86_1", "IC86_2"]):
    try:
        season = copy.copy(ps_v002_p01.seasons[season_name])
    except KeyError:
        season = copy.copy(ps_v002_p01.subseasons[season_name])

    season.season_name = ["IC79-2010", "IC86-2011", "IC86-2012"][i]
    season.sample_name = "all_sky_3_year_mc"

    ps_3_systematic_set.add_season(season)


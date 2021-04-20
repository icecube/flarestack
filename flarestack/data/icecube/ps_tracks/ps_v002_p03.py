"""File containing links to data samples used (pointsource tracks).

Path to local copy of point source tracks, downloaded from /data/ana .. /current
with following README:

    This directory contains an update to version-002p02 which fixes the
    leap second bug for event MJDs in runs 120398 to 126377, inclusive.
    These runs are only in seasons IC86, 2012-2014. See

    https://drive.google.com/file/d/0B6TW2cWERhC6OFFCbWxsTHB1VzQ/view

    For a full description of the leap second issue, which is present
    in level 2 data for both Pass 1 & 2.

    The data files contain the following columns:

      data & MC fields
      ================
      run      -   int64 - run id
      event    -   int64 - event id
      subevent -   int64 - subevent id
        NOTE: Seasons prior to IC86, 2012 have subevent = -1 for all events
              because this info was difficult to find from point source sample
              processing prior to 2012.
      time     - float64 - MJD in days
      ra       - float64 - right ascension in radians
        NOTE: (computed with ra, dec = icecube.astro.dir_to_equa(zen, azi, time))
      dec      - float64 - declination in radians
        NOTE: (computed with ra, dec = icecube.astro.dir_to_equa(zen, azi, time))
      sinDec   - float64 - sin(declination)
      azi      - float64 - azimuth in radians from splineMPE
      zen      - float64 - zenith in radians from splineMPE
      angErr   - float64 - angular error in radians, defined as sigma of 2D Gaussian
        NOTE: angErr is the pull-corrected sigma paraboloid approximated as sigma = sqrt(s1**2 + s2**2)
      logE     - float64 - log10(reco energy/GeV), energy reco is MuEX

      MC only fields
      ==============
      trueE   - float64 - true neutrino energy in GeV
      trueRa  - float64 - true right ascension in radians
      trueDec - float64 - true declination in radians
      ow      - float64 - oneweight in GeV cm2 sr

    File list:

    IC40_exp.npy/IC40_exp.root             Data for IC40 season
    IC59_exp.npy/IC59_exp.root             Data for IC59 season
    IC79_exp.npy/IC79_exp.root             Data for IC79 season
      NOTE: IC79 data derive from the "b" version of the IC79 selection.
            This file was labeled IC79b in version-002p00 and version-002p01
    IC86_2011_exp.npy/IC86_2011_exp.root   Data for IC86, 2011 season
    IC86_2012_exp.npy/IC86_2012_exp.root   Data for IC86, 2012 season
    IC86_2013_exp.npy/IC86_2013_exp.root   Data for IC86, 2013 season
    IC86_2014_exp.npy/IC86_2014_exp.root   Data for IC86, 2014 season

    IC40_MC.npy/IC40_MC.root             Monte Carlo for IC40 season
    IC59_MC.npy/IC59_MC.root             Monte Carlo for IC59 season
    IC79_MC.npy/IC79_MC.root             Monte Carlo for IC79 season
      NOTE: IC79 Monte Carlo derive from the "b" version of the IC79 selection.
            This file was labeled IC79b in version-002p00 and version-002p01
    IC86_2011_MC.npy/IC86_2011_MC.root   Monte Carlo for IC86, 2011 season
    IC86_2012_MC.npy/IC86_2011_MC.root   Monte Carlo for IC86, 2012 season
    IC86_2013_MC.npy/IC86_2011_MC.root   Monte Carlo for IC86, 2013 season
    IC86_2014_MC.npy/IC86_2011_MC.root   Monte Carlo for IC86, 2014 season

    Note that the .npy and .root files have the same events and column names,
    they only differ in their container format.

    Where these files came from?

    These files derive from those under /data/ana/analyses/version-002p01 with
    subevent IDs take from:

    /data/ana/PointSource/IC86_2012_PS/Merged_IC86.2012_*.hd5
    /data/ana/PointSource/IC86_2012_PS/Merged_IC86.2013_*.hd5
    /data/ana/PointSource/IC86_2012_PS/Merged_IC86.2014_*.hd5

    /data/ana/PointSource/IC86_2012_PS/Merged_11*.hd5

"""
from flarestack.data.icecube.ic_season import IceCubeDataset, \
    IceCubeSeason, icecube_dataset_dir
from flarestack.data.icecube.ps_tracks import get_ps_binning
import numpy as np
import logging


logger = logging.getLogger(__name__)

ps_data_dir = icecube_dataset_dir + "ps_tracks/version-002-p03/"
grl_data_dir = ps_data_dir + "GRL/"
ps_v002_p03 = IceCubeDataset()

sample_name = "ps_tracks_v002_p03"
logger.debug(f'building {sample_name}')
logger.debug(f'adding IC40')
ic40 = IceCubeSeason(
    season_name="IC40",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC40_exp.npy",
    mc_path=ps_data_dir + "IC40_MC.npy",
    grl_path=grl_data_dir + "IC40_exp.npy",
    sin_dec_bins=get_ps_binning("IC40")[0],
    log_e_bins=get_ps_binning("IC40")[1]
)

ps_v002_p03.add_season(ic40)

logger.debug('adding IC59')
ic59 = IceCubeSeason(
    season_name="IC59",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC59_exp.npy",
    mc_path=ps_data_dir + "IC59_MC.npy",
    grl_path=grl_data_dir + "IC59_exp.npy",
    sin_dec_bins=get_ps_binning("IC59")[0],
    log_e_bins=get_ps_binning("IC59")[1]
)

ps_v002_p03.add_season(ic59)

logger.debug('adding IC79')
ic79 = IceCubeSeason(
    season_name="IC79",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC79_exp.npy",
    mc_path=ps_data_dir + "IC79_MC.npy",
    grl_path=grl_data_dir + "IC79_exp.npy",
    sin_dec_bins=get_ps_binning("IC79")[0],
    log_e_bins=get_ps_binning("IC79")[1]
)
ps_v002_p03.add_season(ic79)

boundary = np.sin(np.radians(-5.))  # North/South transition boundary

logger.debug('adding IC86 2011')
ic86_1 = IceCubeSeason(
    season_name="IC86_1",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC86_2011_exp.npy",
    mc_path=ps_data_dir + "IC86_2011_MC.npy",
    grl_path=grl_data_dir + "IC86_2011_exp.npy",
    sin_dec_bins=get_ps_binning("IC86_2011")[0],
    log_e_bins=get_ps_binning("IC86_2011")[1]
)

ps_v002_p03.add_season(ic86_1)

# Add optional subseasons for IC86 2, 3, and 4, that can be called instead of
# the combined season

for i in range(2, 5):
    logger.debug(f'adding IC86 201{i}')
    ic86_i = IceCubeSeason(
        season_name="IC86_{0}".format(i),
        sample_name=sample_name,
        exp_path=ps_data_dir + "IC86-201{0}_exp.npy".format(i),
        mc_path=ps_data_dir + "IC86_2012_MC.npy",
        grl_path=grl_data_dir + "IC86_201{0}_exp.npy".format(i),
        sin_dec_bins=get_ps_binning("IC86_2012")[0],
        log_e_bins=get_ps_binning("IC86_2012")[1]
    )
    ps_v002_p03.add_subseason(ic86_i)

logger.debug('adding IC86 2012-2014')
ic86_234 = IceCubeSeason(
    season_name="IC86_234",
    sample_name=sample_name,
    exp_path=[
        ps_data_dir + "IC86_2012_exp.npy",
        ps_data_dir + "IC86_2013_exp.npy",
        ps_data_dir + "IC86_2014_exp.npy"
    ],
    mc_path=ps_data_dir + "IC86_2012_MC.npy",
    grl_path=[
        grl_data_dir + "IC86_2012_exp.npy",
        grl_data_dir + "IC86_2013_exp.npy",
        grl_data_dir + "IC86_2014_exp.npy"
    ],
    sin_dec_bins=get_ps_binning("IC86_2012")[0],
    log_e_bins=get_ps_binning("IC86_2012")[1]
)

ps_v002_p03.add_season(ic86_234)

# ps_3_systematic_set = IceCubeDataset()
#
# for i, season_name in enumerate(["IC79", "IC86_1", "IC86_2"]):
#     try:
#         season = copy.copy(ps_v002_p03.seasons[season_name])
#     except KeyError:
#         season = copy.copy(ps_v002_p03.subseasons[season_name])
#
#     season.season_name = ["IC79-2010", "IC86-2011", "IC86-2012"][i]
#     season.sample_name = "all_sky_3_year_mc"
#
#     ps_3_systematic_set.add_season(season)


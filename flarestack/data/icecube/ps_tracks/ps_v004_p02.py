"""
    This is the interface to the IceCube PS Tracks v4.2 dataset
    For further information, see the readme file contained in the dataset folder.
"""

import numpy as np

from flarestack.data.icecube.ic_season import (
    IceCubeDataset,
    IceCubeSeason,
    icecube_dataset_dir,
)

from flarestack.data.icecube.ps_tracks import get_ps_binning

sample_name = "ps_tracks_v004_p02"

dataset_name = "icecube." + sample_name

dataset_path = "ps_tracks/version-004-p02"

data_path = icecube_dataset_dir / dataset_path

grl_path = data_path / "GRL"


ps_v004_p02 = IceCubeDataset(
    name=dataset_name
)  # instantiate empty dataset to be populated

##########################
#  START ADDING SEASONS  #
##########################

############
# - IC40 - #
############

sinDec_bins = np.unique(
    np.concatenate(
        [
            np.linspace(-1.0, -0.25, 10 + 1),
            np.linspace(-0.25, 0.0, 10 + 1),
            np.linspace(0.0, 1.0, 10 + 1),
        ]
    )
)

energy_bins = np.arange(2.0, 9.0 + 0.01, 0.125)

ic40 = IceCubeSeason(
    season_name="IC40",
    sample_name=sample_name,
    exp_path=data_path / "IC40_exp.npy",
    mc_path=data_path / "IC40_MC.npy",
    grl_path=grl_path / "IC40_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins,
    expect_gaps_in_grl=False,
)

ps_v004_p02.add_season(ic40)

############
# - IC59 - #
############

sinDec_bins = np.unique(
    np.concatenate(
        [
            np.linspace(-1.0, -0.95, 2 + 1),
            np.linspace(-0.95, -0.25, 25 + 1),
            np.linspace(-0.25, 0.05, 15 + 1),
            np.linspace(0.05, 1.0, 10 + 1),
        ]
    )
)

energy_bins = np.arange(2.0, 9.5 + 0.01, 0.125)

ic59 = IceCubeSeason(
    season_name="IC59",
    sample_name=sample_name,
    exp_path=data_path / "IC59_exp.npy",
    mc_path=data_path / "IC59_MC.npy",
    grl_path=grl_path / "IC59_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins,
    expect_gaps_in_grl=False,
)

ps_v004_p02.add_season(ic59)

############
# - IC79 - #
############

sinDec_bins = np.linspace(-1.0, 1.0, 50)
energy_bins = np.arange(2.0, 9.0 + 0.01, 0.125)

ic79 = IceCubeSeason(
    season_name="IC79",
    sample_name=sample_name,
    exp_path=data_path / "IC79_exp.npy",
    mc_path=data_path / "IC79_MC.npy",
    grl_path=grl_path / "IC79_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins,
    expect_gaps_in_grl=False,
)
ps_v004_p02.add_season(ic79)

#######################
# - IC86, 2011-2021 - #
#######################

boundary = np.sin(np.radians(-5.0))  # North/South transition boundary

sinDec_bins = np.unique(
    np.concatenate(
        [
            np.linspace(-1.0, -0.93, 4 + 1),
            np.linspace(-0.93, -0.3, 10 + 1),
            np.linspace(-0.3, 0.05, 9 + 1),
            np.linspace(0.05, 1.0, 18 + 1),
        ]
    )
)

energy_bins = np.arange(1.0, 9.5 + 0.01, 0.125)

mc_path = data_path / "IC86_2016_MC.npy"

IC86_start_year = 2011
IC86_stop_year = 2021
IC86_timerange = range(IC86_start_year, IC86_stop_year + 1)

# full detector configurations have a unified processing now
# so it is not necessary anymore to treat IC86, 2011 differently
for yr in IC86_timerange:
    ic86_i = IceCubeSeason(
        season_name=f"IC86_{yr}",
        sample_name=sample_name,
        exp_path=data_path / f"IC86_{yr}_exp.npy",
        mc_path=mc_path,
        grl_path=grl_path / f"IC86_{yr}_exp.npy",
        sin_dec_bins=sinDec_bins,
        log_e_bins=energy_bins,
        expect_gaps_in_grl=False,
    )
    ps_v004_p02.add_subseason(ic86_i)

# add the combined season

ic86_combo = IceCubeSeason(
    season_name="IC86_1-11",
    sample_name=sample_name,
    exp_path=[data_path / f"IC86_{yr}_exp.npy" for yr in IC86_timerange],
    mc_path=mc_path,
    grl_path=[grl_path / f"IC86_{yr}_exp.npy" for yr in IC86_timerange],
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins,
    expect_gaps_in_grl=False,
)

ps_v004_p02.add_season(ic86_combo)

########################
#  END ADDING SEASONS  #
########################

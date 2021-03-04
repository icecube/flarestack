"""
    This is a first implementation of the PS Tracks combined 10 year dataset

    For changes, see the readme file contained in the
    dataset folder and the presentations here:

    https://events.icecube.wisc.edu/event/125/contributions/7204/attachments/5554/6427/200915_pstracks_updates.pdf
    https://drive.google.com/file/d/1E1I8kgRmlWdLdXX_p7UweXn7gJIYx9Mg/view?usp=sharing
"""

import numpy as np
import copy

from flarestack.data.icecube.ic_season import IceCubeDataset, \
    IceCubeSeason, icecube_dataset_dir
from flarestack.data.icecube.ps_tracks import get_ps_binning


ps_data_dir = icecube_dataset_dir + "ps_tracks/version-004-p00/"
grl_data_dir = ps_data_dir + "GRL/"
ps_v004_p00 = IceCubeDataset()

sample_name = "ps_tracks_v004_p00"

##########################
#  START ADDING SEASONS  #
##########################

############
# - IC40 - #
############

sinDec_bins = np.unique(np.concatenate([
                        np.linspace(-1., -0.25, 10 + 1),
                        np.linspace(-0.25, 0.0, 10 + 1),
                        np.linspace(0.0, 1., 10 + 1),
                        ]))

energy_bins = np.arange(2., 9. + 0.01, 0.125)

ic40 = IceCubeSeason(
    season_name="IC40",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC40_exp.npy",
    mc_path=ps_data_dir + "IC40_MC.npy",
    grl_path=grl_data_dir + "IC40_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins
)

ps_v004_p00.add_season(ic40)

############
# - IC59 - #
############

sinDec_bins = np.unique(np.concatenate([
                        np.linspace(-1., -0.95, 2 + 1),
                        np.linspace(-0.95, -0.25, 25 + 1),
                        np.linspace(-0.25, 0.05, 15 + 1),
                        np.linspace(0.05, 1., 10 + 1),
                        ]))

energy_bins = np.arange(2., 9.5 + 0.01, 0.125)

ic59 = IceCubeSeason(
    season_name="IC59",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC59_exp.npy",
    mc_path=ps_data_dir + "IC59_MC.npy",
    grl_path=grl_data_dir + "IC59_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins
)

ps_v004_p00.add_season(ic59)

############
# - IC79 - #
############

sinDec_bins = np.linspace(-1., 1., 50)
energy_bins = np.arange(2., 9. + 0.01, 0.125)

ic79 = IceCubeSeason(
    season_name="IC79",
    sample_name=sample_name,
    exp_path=ps_data_dir + "IC79_exp.npy",
    mc_path=ps_data_dir + "IC79_MC.npy",
    grl_path=grl_data_dir + "IC79_exp.npy",
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins
)
ps_v004_p00.add_season(ic79)

#######################
# - IC86, 2011-2019 - #
#######################

boundary = np.sin(np.radians(-5.))  # North/South transition boundary

sinDec_bins = np.unique(np.concatenate([
                        np.linspace(-1., -0.93, 4 + 1),
                        np.linspace(-0.93, -0.3, 10 + 1),
                        np.linspace(-0.3, 0.05, 9 + 1),
                        np.linspace(0.05, 1., 18 + 1),
                        ]))

energy_bins = np.arange(1., 9.5 + 0.01, 0.125)

mc_path = ps_data_dir + "IC86_2016_MC.npy"

# full detector configurations have a unified processing now
# so it is not necessary anymore to treat IC86, 2011 differently
for i in range(1, 10):
    ic86_i = IceCubeSeason(
        season_name="IC86_{0}".format(i),
        sample_name=sample_name,
        exp_path=ps_data_dir + "IC86_201{0}_exp.npy".format(i),
        mc_path=mc_path,
        grl_path=grl_data_dir + "IC86_201{0}_exp.npy".format(i),
        sin_dec_bins=sinDec_bins,
        log_e_bins=energy_bins
    )
    ps_v004_p00.add_subseason(ic86_i)

# add the combined season

ic86_123456789 = IceCubeSeason(
    season_name="IC86_1-9",
    sample_name=sample_name,
    exp_path=[f"{ps_data_dir}IC86_201{i}_exp.npy" for i in range(1, 10)],
    mc_path=mc_path,
    grl_path=[f"{grl_data_dir}IC86_201{i}_exp.npy" for i in range(1, 10)],
    sin_dec_bins=sinDec_bins,
    log_e_bins=energy_bins
)

ps_v004_p00.add_season(ic86_123456789)

########################
#  END ADDING SEASONS  #
########################

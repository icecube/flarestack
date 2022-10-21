from flarestack.data.icecube.ic_season import IceCubeDataset, icecube_dataset_dir
from flarestack.data.icecube.northern_tracks import (
    NTSeasonNewStyle,
    get_diffuse_binning,
)

nt_data_dir = icecube_dataset_dir / "northern_tracks/version-005-p01"

sample_name = "northern_tracks_v005_p01"

dataset_name = "icecube." + sample_name

nt_v005_p01 = IceCubeDataset(name=dataset_name)

IC86_start_year = 2011
IC86_stop_year = 2021
IC86_timerange = range(IC86_start_year, IC86_stop_year + 1)

seasons = [f"IC86_{yr}" for yr in IC86_timerange]

# ==================================
# Add individual years as subseasons
# ==================================


def generate_diffuse_season(name):
    season = NTSeasonNewStyle(
        season_name=name,
        sample_name=sample_name,
        exp_path=nt_data_dir / f"{name}_exp.npy",
        mc_path=nt_data_dir / "IC86_pass2_MC.npy",
        grl_path=nt_data_dir / f"GRL/{name}_exp.npy",
        sin_dec_bins=get_diffuse_binning(name)[0],
        log_e_bins=get_diffuse_binning(name)[1],
    )
    return season


for season in seasons:
    subseason = generate_diffuse_season(season)
    nt_v005_p01.add_subseason(subseason)


# ==================================
# Add combo season
# ==================================

name = "IC86_1-11"

combo_season = NTSeasonNewStyle(
    season_name=name,
    sample_name=sample_name,
    exp_path=[nt_data_dir / f"IC86_{yr}_exp.npy" for yr in IC86_timerange],
    mc_path=nt_data_dir / "IC86_pass2_MC.npy",
    grl_path=[nt_data_dir / f"GRL/IC86_{yr}_exp.npy" for yr in IC86_timerange],
    sin_dec_bins=get_diffuse_binning(name)[0],
    log_e_bins=get_diffuse_binning(name)[1],
)

nt_v005_p01.add_season(combo_season)

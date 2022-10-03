from flarestack.data.icecube.ic_season import IceCubeDataset, get_dataset_dir
from flarestack.data.icecube.northern_tracks import (
    NTSeasonNewStyle,
    get_diffuse_binning,
)

icecube_dataset_dir = get_dataset_dir()

from flarestack.data.dataset_index import dataset_index

nt_data_dir = icecube_dataset_dir + "northern_tracks/version-005-p01/"

nt_v005_p01 = IceCubeDataset()

sample_name = "northern_tracks_v005_p01"


def generate_diffuse_season(name):
    season = NTSeasonNewStyle(
        season_name=name,
        sample_name=sample_name,
        exp_path=nt_data_dir + f"{name}_exp.npy",
        mc_path=nt_data_dir + "IC86_pass2_MC.npy",
        grl_path=nt_data_dir + f"GRL/{name}_exp.npy",
        sin_dec_bins=get_diffuse_binning(name)[0],
        log_e_bins=get_diffuse_binning(name)[1],
    )
    nt_v005_p01.add_season(season)


IC86_start_year = 2011
IC86_stop_year = 2021
IC86_timerange = range(IC86_start_year, IC86_stop_year + 1)

seasons = [f"IC86_{yr}" for yr in IC86_timerange]

for season in seasons:
    generate_diffuse_season(season)

dataset_index.add_dataset("icecube." + sample_name, nt_v005_p01)

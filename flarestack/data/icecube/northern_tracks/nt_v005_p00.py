from flarestack.data.icecube.ic_season import IceCubeDataset, icecube_dataset_dir
from flarestack.data.icecube.northern_tracks import (
    NTSeasonNewStyle,
    get_diffuse_binning,
)


nt_data_dir = icecube_dataset_dir + "northern_tracks/version-005-p00/"

nt_v005_p00 = IceCubeDataset()

sample_name = "northern_tracks_v005_p00"


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
    nt_v005_p00.add_season(season)


seasons = [f"IC86_201{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]]

for season in seasons:
    generate_diffuse_season(season)

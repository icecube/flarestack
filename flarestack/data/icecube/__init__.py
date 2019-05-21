from flarestack.data import Dataset, SeasonWithMC
from flarestack.icecube_utils.dataset_loader import data_loader, grl_loader, \
    convert_grl, verify_grl_with_data


class IceCubeDataset(Dataset):
    pass


class IceCubeSeason(SeasonWithMC):

    def __init__(self, season_name, sample_name, exp_path, mc_path, grl_path,
                 sin_dec_bins, log_e_bins, **kwargs):
        SeasonWithMC.__init__(self, season_name, sample_name, exp_path, mc_path,
                              **kwargs)
        self.grl_path = grl_path
        self.all_paths.append(grl_path)
        self.sin_dec_bins = sin_dec_bins
        self.log_e_bins = log_e_bins

    def get_livetime_data(self):
        return convert_grl(self)

    def check_data_quality(self):
        verify_grl_with_data(self)

    def get_grl(self):
        return grl_loader(self)

    def load_data(self, path, **kwargs):
        return data_loader(path, **kwargs)

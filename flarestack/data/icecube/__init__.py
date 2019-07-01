import numpy as np
import os
from flarestack.data import Dataset, SeasonWithMC
from flarestack.icecube_utils.dataset_loader import data_loader, grl_loader, \
    convert_grl, verify_grl_with_data
from flarestack.shared import host_server

if host_server == "DESY":
    icecube_dataset_dir = "/lustre/fs22/group/icecube/data_mirror/"
    skylab_ref_dir = icecube_dataset_dir + "mirror-7year-PS-sens/"
    print("Loading datasets from", icecube_dataset_dir, "(DESY)")
elif host_server == "WIPAC":
    icecube_dataset_dir = "/data/ana/analyses/"
    skylab_ref_dir = "/data/user/steinrob/mirror-7year-PS-sens/"
    print("Loading datasets from", icecube_dataset_dir, "(WIPAC)")
else:
    icecube_dataset_dir = None


# Dataset directory can be changed if needed

def set_icecube_dataset_directory(path):
    """Sets the dataset directory to be a custom path, and exports this.

    :param path: Path to datasets
    """
    if not os.path.isdir(path):
        raise Exception("Attempting to set invalid path for datasets. "
                        "Directory", path, "does not exist!")
    print("Loading datasets from", path)

    global icecube_dataset_dir
    icecube_dataset_dir = path


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

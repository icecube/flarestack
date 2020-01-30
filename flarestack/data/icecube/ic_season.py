import numpy as np
import os
from flarestack.data import Dataset, SeasonWithMC
from flarestack.icecube_utils.dataset_loader import data_loader, grl_loader, \
    verify_grl_with_data
from flarestack.shared import host_server
from flarestack.core.time_pdf import TimePDF, DetectorOnOffList
from scipy.interpolate import interp1d
import logging

try:
    icecube_dataset_dir = os.environ['FLARESTACK_DATASET_DIR']

    if os.path.isdir(icecube_dataset_dir + "mirror-7year-PS-sens/"):
        published_sens_ref_dir = icecube_dataset_dir + "mirror-7year-PS-sens/"
    logging.info(f"Loading datasets from {icecube_dataset_dir} (local)")
except KeyError:
    icecube_dataset_dir = None

if icecube_dataset_dir is None:
    if host_server == "DESY":
        icecube_dataset_dir = "/lustre/fs22/group/icecube/data_mirror/"
        published_sens_ref_dir = icecube_dataset_dir + "mirror-7year-PS-sens/"
        logging.info(f"Loading datasets from {icecube_dataset_dir} (DESY)")
    elif host_server == "WIPAC":
        icecube_dataset_dir = "/data/ana/analyses/"
        published_sens_ref_dir = "/data/user/steinrob/mirror-7year-PS-sens/"
        logging.info(f"Loading datasets from {icecube_dataset_dir} (WIPAC)")
    else:
        raise ImportError("No IceCube data directory found. Run: \n"
                          "export FLARESTACK_DATA_DIR=/path/to/IceCube/data")

def get_published_sens_ref_dir():
    try:
        return published_sens_ref_dir
    except NameError:
        logging.error(
            "No reference sensitivity directory found. "
            "Please create one at {0}".format(
            icecube_dataset_dir + "mirror-7year-PS-sens/"
            ))
        raise

# # Dataset directory can be changed if needed
#
# def set_icecube_dataset_directory(path):
#     """Sets the dataset directory to be a custom path, and exports this.
#
#     :param path: Path to datasets
#     """
#     if not os.path.isdir(path):
#         raise Exception("Attempting to set invalid path for datasets. "
#                         "Directory", path, "does not exist!")
#     print("Loading datasets from", path)
#
#     global icecube_dataset_dir
#     icecube_dataset_dir = path

@TimePDF.register_subclass("icecube_on_off_list")
class IceCubeRunList(DetectorOnOffList):
    """Custom TimePDF class designed to constructed a pdf from an IceCube
    GoodRunList.
    """

    def parse_list(self):

        if list(self.on_off_list["run"]) != sorted(list(self.on_off_list["run"])):
            logging.error("Error in ordering GoodRunList!")
            logging.error("Runs are out of order!")
            self.on_off_list = np.sort(self.on_off_list, order="run")

        t0 = min(self.on_off_list["start"])
        t1 = max(self.on_off_list["stop"])

        full_livetime = np.sum(self.on_off_list["length"])

        step = 1e-10

        t_range = [t0 - step]
        f = [0.]

        mjd = [0.]
        livetime = [0.]
        total_t = 0.

        for i, run in enumerate(self.on_off_list):
            mjd.append(run["start"])
            livetime.append(total_t)
            total_t += run["length"]
            mjd.append(run["stop"])
            livetime.append(total_t)

            t_range.extend([
                run["start"] - step, run["start"], run["stop"],
                run["stop"] + step
            ])
            f.extend([0., 1., 1., 0.])

        stitch_t = t_range
        stitch_f = f

        if stitch_t != sorted(stitch_t):
            logging.error("Error in ordering GoodRunList!")
            logging.error("Runs are out of order!")

        mjd.append(1e5)
        livetime.append(total_t)

        season_f = interp1d(stitch_t, np.array(stitch_f), kind="linear")
        mjd_to_livetime = interp1d(mjd, livetime, kind="linear")
        livetime_to_mjd = interp1d(livetime, mjd, kind="linear")
        return t0, t1, full_livetime, season_f, mjd_to_livetime, livetime_to_mjd


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

    # def get_livetime_data(self):
    #     return convert_grl(self)

    def check_data_quality(self):
        verify_grl_with_data(self)

    def get_grl(self):
        return grl_loader(self)

    def load_data(self, path, **kwargs):
        return data_loader(path, **kwargs)

    def build_time_pdf_dict(self):
        """Function to build a pdf for the livetime of the season. By
        default, this is assumed to be uniform, spanning from the first to
        the last event found in the data.

        :return: Time pdf dictionary
        """

        t_pdf_dict = {
            "time_pdf_name": "icecube_on_off_list",
            "on_off_list": self.get_grl()
        }

        return t_pdf_dict

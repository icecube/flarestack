import numpy as np
import os
from flarestack.data import Dataset, SeasonWithMC
from flarestack.icecube_utils.dataset_loader import data_loader, grl_loader, \
    verify_grl_with_data
from flarestack.shared import host_server
from flarestack.core.time_pdf import TimePDF, DetectorOnOffList
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

try:
    icecube_dataset_dir = os.environ['FLARESTACK_DATASET_DIR']

    if os.path.isdir(icecube_dataset_dir + "mirror-7year-PS-sens/"):
        ref_dir_7yr = icecube_dataset_dir + "mirror-7year-PS-sens/"
    logger.info(f"Loading datasets from {icecube_dataset_dir} (local)")
except KeyError:
    icecube_dataset_dir = None

if icecube_dataset_dir is None:
    if host_server == "DESY":
        icecube_dataset_dir = "/lustre/fs22/group/icecube/data_mirror/"
        ref_dir_7yr = icecube_dataset_dir + "ref_sensitivity/mirror-7year-PS-sens/"
        ref_10yr = icecube_dataset_dir + "ref_sensitivity/TenYr_E2andE3_sensitivity_and_discpot.npy"
        logger.info(f"Loading datasets from {icecube_dataset_dir} (DESY)")
    elif host_server == "WIPAC":
        icecube_dataset_dir = "/data/ana/analyses/"
        ref_dir_7yr = "/data/ana/PointSource/PS/version-002-p01/results/time_integrated_fullsky/"
        ref_10yr = "/data/user/tcarver/skylab_scripts/skylab_trunk/doc/analyses/combined_tracks/TenYr_E2andE3_sensitivity_and_discpot.npy"
        logger.info(f"Loading datasets from {icecube_dataset_dir} (WIPAC)")
    else:
        raise ImportError("No IceCube data directory found. Run: \n"
                          "export FLARESTACK_DATA_DIR=/path/to/IceCube/data")

def get_published_sens_ref_dir():
    try:
        return ref_dir_7yr, ref_10yr
    except NameError:
        logger.error(
            "No reference sensitivity directory found. "
            "Please create one at {0}".format(
            icecube_dataset_dir + "mirror-7year-PS-sens/"
            ))
        raise

@TimePDF.register_subclass("icecube_on_off_list")
class IceCubeRunList(DetectorOnOffList):
    """Custom TimePDF class designed to constructed a pdf from an IceCube
    GoodRunList.
    """

    def parse_list(self):

        if list(self.on_off_list["run"]) != sorted(list(self.on_off_list["run"])):
            logger.error("Error in ordering GoodRunList!")
            logger.error("Runs are out of order!")
            self.on_off_list = np.sort(self.on_off_list, order="run")

        mask = self.on_off_list["stop"][:-1] == self.on_off_list["start"][1:]

        if np.sum(mask) > 0:

            first_run = self.on_off_list["run"][:-1][mask][0]

            logger.error("The IceCube GoodRunList was not produced correctly.")
            logger.error("Some runs in the GoodRunList start immediately after the preceding run ends.")
            logger.error("There should be gaps between every run due to detector downtime, but some are missing here.")
            logger.error(f"The first missing gap is between runs {first_run} and {first_run+1}.")
            logger.error("Any livetime estimates using this GoodRunList will not be accurate.")
            logger.error("This is a known problem affecting older IceCube GoodRunLists.")
            logger.error("You should use a newer, corrected GoodRunList.")
            logger.error("Flarestack will attempt to stitch these runs together.")
            logger.error("However, livetime estimates may be off by several percentage points, "
                          "or even more for very short timescales.")
            logger.error("You have been warned!")

            while np.sum(mask) > 0:

                index = list(mask).index(True)

                self.on_off_list[index]["stop"] = self.on_off_list[index+1]["stop"]
                self.on_off_list[index]["length"] += self.on_off_list[index+1]["length"]
                self.on_off_list[index]["events"] += self.on_off_list[index + 1]["events"]

                mod_mask = np.arange(len(self.on_off_list)) == index+1

                self.on_off_list = self.on_off_list[~mod_mask]

                mask = self.on_off_list["stop"][:-1] == self.on_off_list["start"][1:]

        mask = self.on_off_list["stop"][:-1] < self.on_off_list["start"][1:]

        if np.sum(~mask) > 0:

            first_run = self.on_off_list["run"][:-1][~mask][0]

            logger.error("The IceCube GoodRunList was not produced correctly.")
            logger.error("Some runs in the GoodRunList start before the preceding run has ended.")
            logger.error("Under no circumstances should runs overlap.")
            logger.error(f"The first overlap is between runs {first_run} and {first_run+1}.")
            logger.error("Any livetime estimates using this GoodRunList will not be accurate.")
            logger.error("This is a known problem affecting older IceCube GoodRunLists.")
            logger.error("You should use a newer, corrected GoodRunList.")
            logger.error("Flarestack will attempt to stitch these runs together.")
            logger.error("However, livetime estimates may be off by several percentage points, "
                          "or even more for very short timescales.")
            logger.error("You have been warned!")

            while np.sum(~mask) > 0:

                index = list(~mask).index(True)

                self.on_off_list[index]["stop"] = self.on_off_list[index+1]["stop"]
                self.on_off_list[index]["length"] += self.on_off_list[index+1]["length"]
                self.on_off_list[index]["events"] += self.on_off_list[index + 1]["events"]

                mod_mask = np.arange(len(self.on_off_list)) == index+1

                self.on_off_list = self.on_off_list[~mod_mask]

                mask = self.on_off_list["stop"][:-1] < self.on_off_list["start"][1:]

        t0 = min(self.on_off_list["start"])
        t1 = max(self.on_off_list["stop"])

        full_livetime = np.sum(self.on_off_list["length"])

        step = 1e-12

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
            logger.error("Error in ordering GoodRunList somehow!")
            logger.error("Runs are out of order!")

            for i, t in enumerate(stitch_t):
                if t != sorted(stitch_t)[i]:
                    print(t, sorted(stitch_t)[i])
                    print(stitch_t[i-1:i+2])
                    print(sorted(stitch_t)[i-1:i+2])
                    key = int((i-1)/4)
                    print(self.on_off_list[key:key+2])
                    input("????")
            
            raise Exception(f"Runs in GoodRunList are out of order for {self.on_off_list}. Check that!")

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

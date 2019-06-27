import flarestack.data
import os
import numpy as np
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.shared import sim_dataset_dir_path


def generate_sim_season_class(season_class, **kwargs):

    if not issubclass(season_class, flarestack.data.SeasonWithoutMC):
        raise Exception("{0} is not a subclass of SeasonWithoutMC. Cannot "
                        "simulate data without effective areas!")

    base_args = kwargs

    class SimSeason(season_class):

        def __init__(self, mjd_start, mjd_end, bkg_flux_norm, bkg_e_pdf_dict,
                     **kwargs):

            season_class.__init__(self, **base_args)
            self.mjd_start = float(mjd_start)
            self.mjd_end = float(mjd_end)
            self.bkg_flux_norm = bkg_flux_norm
            self.bkg_energy_pdf = EnergyPDF(bkg_e_pdf_dict)
            self.base_dataset_path = sim_dataset_dir_path(
                self, bkg_flux_norm, bkg_e_pdf_dict)
            self.exp_path = self.check_data_period()


        def dataset_path(self, mjd_start=None, mjd_end=None):
            try:
                os.makedirs(self.base_dataset_path)
            except OSError:
                pass

            if mjd_start is None:
                mjd_start = self.mjd_start

            if mjd_end is None:
                mjd_end = self.mjd_end

            return "{0}/{1}_{2}.npy".format(
                self.base_dataset_path, mjd_start, mjd_end)

        def load_data(self, path, **kwargs):
            data = season_class.load_data(path, kwargs)
            mask = np.logical_and(
                data["time"] > self.mjd_start,
                data["time"] < self.mjd_end
            )
            return data[mask]

        def check_data_period(self):
            if not os.path.isfile(self.dataset_path()):
                data = self.simulate(self.mjd_start, self.mjd_end)
                np.save(self.dataset_path(), data)
            else:
                filename = os.path.basename(self.dataset_path())
                print(self.mjd_start, self.mjd_end)
                base_start, base_end = filename.split("_")
                base_end = base_end[:-4]
                print(filename)
                print(base_start, base_end)
                print(self.get_background_model())

            os.remove(self.dataset_path())
            return self.dataset_path()

        @staticmethod
        def simulate(mjd_start, mjd_end, old=None):
            return old

    return SimSeason

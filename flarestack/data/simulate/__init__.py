from flarestack.data import SeasonWithoutMC, Dataset
import os
import numpy as np
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.shared import sim_dataset_dir_path, k_to_flux


# def generate_sim_season_class(season_class, **kwargs):
#
#     if not issubclass(season_class, flarestack.data.SeasonWithoutMC):
#         raise Exception("{0} is not a subclass of SeasonWithoutMC. Cannot "
#                         "simulate data without effective areas!")
#
#     base_args = kwargs

sim_dir = os.path.abspath(os.path.dirname(__file__))
raw_sim_data_dir = sim_dir + "/raw_data/"

class SimDataset(Dataset):

    def __init__(self, **kwargs):
        Dataset.__init__(self, **kwargs)
        self.sim_seasons = dict()
        self.init_args = dict()

    def add_season(self, season):
        raise Exception("SimDatasets are not compatible add_season function. "
                        "You must add SimSeason classes using add_sim_season, "
                        "and then set the simulation period for each season "
                        "using set_sim_period. ")

    def add_subseason(self, season):
        self.add_season(season)

    def add_sim_season(self, name, sim_season_f):
        self.sim_seasons[name] = sim_season_f

    def set_sim_params(self, name, bkg_time_pdf_dict, bkg_flux_norm,
                       bkg_e_pdf_dict, **kwargs):

        if name not in self.sim_seasons.keys():
            raise KeyError("SimSeasonClass {0} not found. The following "
                           "SimSeasonClasses were found {1}. Additional "
                           "classes must be added with the add_sim_season "
                           "function.".format(name, self.sim_seasons.keys()))


        self.seasons[name] = self.sim_seasons[name](
            bkg_time_pdf_dict, bkg_flux_norm, bkg_e_pdf_dict, **kwargs)


class SimSeason(SeasonWithoutMC):

    def __init__(self, season_name, sample_name, pseudo_mc_path,
                 event_dtype, effective_area_f, load_angular_resolution,
                 bkg_t_pdf_dict, bkg_flux_norm, bkg_e_pdf_dict,
                 energy_proxy_map, **kwargs):

        self.bkg_t_pdf_dict = bkg_t_pdf_dict
        self.event_dtype = event_dtype

        self.bkg_flux_norm = bkg_flux_norm
        self.bkg_e_pdf_dict = bkg_e_pdf_dict

        self.load_angular_resolution = load_angular_resolution
        self.angular_res_f = self.load_angular_resolution()

        self.base_dataset_path = sim_dataset_dir_path(
            sample_name, season_name, bkg_flux_norm, bkg_e_pdf_dict)

        exp_path = self.dataset_path()

        SeasonWithoutMC.__init__(
            self, season_name, sample_name, exp_path, pseudo_mc_path, **kwargs
        )
        self.bkg_energy_pdf = EnergyPDF.create(bkg_e_pdf_dict)

        self.energy_proxy_map = energy_proxy_map

        self.effective_area_f = effective_area_f

        self.check_sim(**kwargs)

    def load_energy_proxy_mapping(self):
        return self.energy_proxy_map

    def build_time_pdf_dict(self):
        return self.bkg_t_pdf_dict

    def check_sim(self, resimulate=False, **kwargs):

        if np.logical_and(not resimulate, os.path.isfile(self.exp_path)):
            print("Found existing simulation at {0}".format(self.exp_path))
        else:
            self.simulate()

    def dataset_path(self, mjd_start=None, mjd_end=None):

        time_pdf = self.build_time_pdf()

        try:
            os.makedirs(self.base_dataset_path)
        except OSError:
            pass

        if mjd_start is None:
            mjd_start = time_pdf.sig_t0([])

        if mjd_end is None:
            mjd_end = time_pdf.sig_t1([])

        return "{0}/{1}_{2}.npy".format(
            self.base_dataset_path, mjd_start, mjd_end)

    def simulate(self):
        ti_flux = self.get_time_integrated_flux()
        sim_data = self.generate_sim_data(ti_flux)
        np.save(self.exp_path, sim_data)

    def generate_sim_data(self, fluence):
        raise NotImplementedError(
            "No generate_sim_data function has been implemented for "
            "class {0}".format(self.__class__.__name__))
        return

    def get_time_integrated_flux(self):
        return k_to_flux(
            self.bkg_flux_norm * self.get_time_pdf().effective_injection_time())



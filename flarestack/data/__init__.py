"""Generic classes for a Dataset, and for a Season.
"""
import os
import numpy as np
from numpy.lib.recfunctions import append_fields, drop_fields
from flarestack.core.injector import MCInjector, EffectiveAreaInjector
from flarestack.utils.make_SoB_splines import make_background_spline
from flarestack.utils.create_acceptance_functions import make_acceptance_season



class Dataset:
    def __init__(self, **kwargs):
        self.seasons = dict()
        self.subseasons = dict()

    def add_season(self, season):
        self.seasons[season.season_name] = season

    def add_subseason(self, season):
        self.subseasons[season.season_name] = season

    def get_seasons(self, *args):
        season_names = list(args)
        if len(season_names) == 0:
            return dict(self.seasons)
        else:
            season_dict = dict()
            for name in season_names:
                if name in self.seasons:
                    season_dict[name] = self.seasons[name]
                elif name in self.subseasons:
                    season_dict[name] = self.subseasons[name]
                else:
                    raise Exception(
                        "Unrecognised season name: {0} not found".format(name))
            return season_dict


class Season:

    def __init__(self, season_name, sample_name, exp_path, **kwargs):
        self.season_name = season_name
        self.sample_name = sample_name
        self.exp_path = exp_path
        self.loaded_background = None
        self.pseudo_mc_path = None
        self.all_paths = [self.exp_path]

    def load_background_data(self):
        """Generic function to load background data to memory. It is useful
        for Injector, but does not always need to be used."""
        self.loaded_background = self.get_background_model()

    def get_background_dtype(self):
        return drop_fields(self.loaded_background, "weight").dtype

    def get_background_model(self, **kwargs):
        """Generic Function to return background model. This could be
        the experimental data (if the signal contamination is small),
        or a weighted MC dataset."""
        exp = self.get_exp_data(**kwargs)
        weight = np.ones(len(exp))
        exp = append_fields(
            exp, 'weight', weight, usemask=False, dtypes=[np.float]
        ).copy()
        return exp

    def pseudo_background(self):
        """Scrambles the raw dataset to "blind" the data. Assigns a flat Right
        Ascension distribution, and randomly redistributes the arrival times
        in the dataset. Returns a shuffled dataset, which can be used for
        blinded analysis.
        :return: data: The scrambled dataset
        """
        data = np.copy(self.loaded_background)
        # Assigns a flat random distribution for Right Ascension
        data['ra'] = np.random.uniform(0, 2 * np.pi, size=len(data))
        # Randomly reorders the times
        np.random.shuffle(data["time"])
        return data[list(self.get_background_dtype().names)].copy()

    def get_exp_data(self, **kwargs):
        return self.load_data(self.exp_path, **kwargs)

    def get_livetime_data(self):
        pass

    def load_data(self, path, **kwargs):
        return np.load(path)

    def make_injector(self, sources, **inj_kwargs):
        pass

    def return_name(self):
        return self.sample_name + "/" + self.season_name

    def make_background_spatial(self):
        make_background_spline(self)

    def get_pseudo_mc(self, **kwargs):
        return self.load_data(self.pseudo_mc_path, **kwargs)

    def check_files_exist(self):

        all_paths = []

        for x in self.all_paths:
            if isinstance(x, list):
                all_paths += x
            else:
                all_paths.append(x)

        for x in all_paths:
            if not os.path.isfile(x):
                raise Exception("File Not Found: {0}".format(x))
            else:
                print("Found:", x)

    def check_data_quality(self):
        pass


    # def make_acceptance_function(self, acc_path):
    #     make_acceptance_season(self, acc_path)


class SeasonWithMC(Season):

    def __init__(self, season_name, sample_name, exp_path, mc_path, **kwargs):
        Season.__init__(self, season_name, sample_name, exp_path, **kwargs)
        self.mc_path = mc_path
        self.all_paths.append(self.mc_path)
        self.pseudo_mc_path = mc_path

    def make_injector(self, sources, **inj_kwargs):
        return MCInjector.create(self, sources, **inj_kwargs)

    def get_mc(self, **kwargs):
        return self.load_data(self.mc_path, **kwargs)


class SeasonWithoutMC(Season):

    def __init__(self, season_name, sample_name, exp_path, pseudo_mc_path,
                 **kwargs):
        Season.__init__(self, season_name, sample_name, exp_path, **kwargs)
        self.pseudo_mc_path = pseudo_mc_path
        self.all_paths.append(self.pseudo_mc_path)

    def make_injector(self, sources, **inj_kwargs):
        return EffectiveAreaInjector.create(self, sources, **inj_kwargs)

    def load_effective_area(self):
        return

    def load_angular_resolution(self):
        return

    def load_energy_proxy_mapping(self):
        """Function to construct the mapping between energy proxy and true
        energy. By default, this is a simple 1:1 mapping.

        :return: function mapping true energy to energy proxy
        """
        return lambda x: np.log10(x)

    def get_livetime_data(self):
        exp = self.load_data(self.exp_path)
        t0 = min(exp["time"])
        t1 = max(exp["time"])
        livetime = t1 - t0
        season_f = lambda x: 1.
        mjd_to_livetime = lambda x: x - t0
        livetime_to_mjd = lambda x: x + t0

        return t0, t1, livetime, season_f, mjd_to_livetime, livetime_to_mjd
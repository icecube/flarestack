import numpy as np
import math
import os
import zlib
from flarestack.core.energy_PDFs import EnergyPDF
from flarestack.shared import min_angular_err, base_floor_quantile, \
    floor_pickle, pull_pickle, weighted_quantile
from flarestack.utils.dynamic_pull_correction import \
    create_quantile_floor_0d, create_quantile_floor_0d_e, \
    create_quantile_floor_1d, create_quantile_floor_1d_e, create_pull_0d_e
from flarestack.utils.dataset_loader import data_loader
import json
import cPickle as Pickle
from scipy.interpolate import interp1d, RectBivariateSpline


class BaseFloorClass:
    subclasses = {}

    def __init__(self, floor_dict):
        self.floor_dict = floor_dict
        self.season = floor_dict["season"]
        self.pickle_name = floor_pickle(floor_dict)

    @classmethod
    def register_subclass(cls, floor_name):
        """Adds a new subclass of BaseFloorClass, with class name equal to
        "floor_name".
        """

        def decorator(subclass):
            cls.subclasses[floor_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, floor_dict):

        floor_name = floor_dict["floor_name"]

        if floor_name not in cls.subclasses:
            raise ValueError('Bad floor name {}'.format(floor_name))

        return cls.subclasses[floor_name](floor_dict)

    def floor(self, data):
        return np.array([0. for _ in data])

    def apply_floor(self, data):
        data[data["raw_sigma"] < self.floor(data)]["sigma"] = math.sqrt(
            self.floor(data) ** 2. + data["raw_sigma"] ** 2.)
        return data

    def apply_dynamic(self, data):
        return data

    def apply_static(self, data):
        return data


@BaseFloorClass.register_subclass('no_floor')
class NoFloor(BaseFloorClass):
    pass


class BaseStaticFloor(BaseFloorClass):
    """Class that enables the application of a static floor. Rewrites the
    apply_static method to update the "sigma" field AND update the "raw
    sigma" field. This means that the floor is applied once, and then is
    permanently in effect. It will run faster because it will not be
    reapplied in each iteration.
    """

    def apply_static(self, data):
        data = self.floor(data)
        data["raw_sigma"] = data["sigma"]
        return data


class BaseDynamicFloorClass(BaseFloorClass):
    """Class that enables the application of a dynamic floor. Rewrites the
    apply_dynamic method to update only the "sigma" field". This means that
    the floor is applied for each iteration. It will run slower because it will
    be reapplied in each iteration.
    """

    def apply_dynamic(self, data):
        return self.floor(data)


@BaseFloorClass.register_subclass('static_floor')
class StaticFloor(BaseStaticFloor):

    def __init__(self, floor_dict):
        BaseFloorClass.__init__(self, floor_dict)

        try:
            self.min_error = np.deg2rad(floor_dict["min_error_deg"])
        except KeyError:
            self.min_error = min_angular_err

        print "Applying an angular error floor of", np.degrees(self.min_error),
        print "degrees"

    def floor(self, data):
        return np.array([self.min_error for _ in data])


class BaseQuantileFloor(BaseFloorClass):

    def __init__(self, floor_dict):

        try:
            self.floor_quantile = floor_dict["floor_quantile"]
        except KeyError:
            self.floor_quantile = base_floor_quantile
            floor_dict["floor_quantile"] = self.floor_quantile

        BaseFloorClass.__init__(self, floor_dict)

        print "Applying an angular error floor using quantile",
        print self.floor_quantile

        if not os.path.isfile(self.pickle_name):
            self.create_pickle()
        else:
            print "Loading from", self.pickle_name

        with open(self.pickle_name, "r") as f:
            pickled_data = Pickle.load(f)

        self.f = self.create_function(pickled_data)

    def create_pickle(self):
        pass

    def create_function(self, pickled_array):
        pass


@BaseFloorClass.register_subclass('quantile_floor_0d')
class QuantileFloor0D(BaseQuantileFloor, BaseStaticFloor):

    def create_pickle(self):
        create_quantile_floor_0d(self.floor_dict)

    def create_function(self, pickled_array):
        return lambda data, params: np.array([pickled_array for _ in data])



@BaseFloorClass.register_subclass('quantile_floor_0d_e')
class QuantileFloorEParam0D(BaseQuantileFloor, BaseDynamicFloorClass):

    def create_pickle(self):
        create_quantile_floor_0d_e(self.floor_dict)

    def create_function(self, pickled_array):
        print pickled_array
        func = interp1d(pickled_array[0], pickled_array[1])
        return lambda data, params: np.array([func(params) for _ in data])
#
@BaseFloorClass.register_subclass('quantile_floor_1d')
class QuantileFloor1D(BaseQuantileFloor, BaseStaticFloor):

    def create_pickle(self):
        create_quantile_floor_1d(self.floor_dict)

    def create_function(self, pickled_array):
        func = interp1d(pickled_array[0], pickled_array[1])
        return lambda data, params: func(data["logE"])

@BaseFloorClass.register_subclass('quantile_floor_1d_e')
class QuantileFloor1D(BaseQuantileFloor, BaseDynamicFloorClass):

    def create_pickle(self):
        create_quantile_floor_1d_e(self.floor_dict)

    def create_function(self, pickled_array):
        func = RectBivariateSpline(
            pickled_array[0], pickled_array[1],
            np.log(pickled_array[2]),
            kx=1, ky=1, s=0)
        return lambda data, params: np.array(
            [np.exp(func(x["logE"], params[0])[0]) for x in data]).T


class BasePullCorrector:
    subclasses = {}

    def __init__(self, pull_dict):
        self.season = pull_dict["season"]
        self.floor = BaseFloorClass.create(pull_dict)
        self.pull_dict = pull_dict
        self.pull_name = pull_pickle(pull_dict)


    @classmethod
    def register_subclass(cls, pull_name):
        """Adds a new subclass of BasePullCorrector, with class name equal to
        "pull_name".
        """

        def decorator(subclass):
            cls.subclasses[pull_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, season, e_pdf_dict, floor_name="static_floor",
               pull_name="no_pull", **kwargs):

        pull_dict = dict()
        pull_dict["season"] = season
        pull_dict["e_pdf_dict"] = e_pdf_dict
        pull_dict["floor_name"] = floor_name
        pull_dict["pull_name"] = pull_name
        pull_dict.update(kwargs)

        if pull_name not in cls.subclasses:
            raise ValueError('Bad pull name {}'.format(pull_name))

        return cls.subclasses[pull_name](pull_dict)

    def pull_correct(self, data):
        data["sigma"] = self.f(data["raw_sigma"])
        return data

    def pull_correct_static(self, data):
        data = self.floor.apply_static(data)
        return data

    def pull_correct_dynamic(self, data, **kwargs):
        data = self.floor.apply_dynamic(data, **kwargs)
        return data


@BasePullCorrector.register_subclass('no_pull')
class NoPull(BasePullCorrector):
    pass


class BaseStaticPullCorrector(BasePullCorrector):

    def pull_correct_static(self, data):
        data = self.floor.apply_static(data)
        data = self.pull_correct(data)
        data["raw_sigma"] = data["sigma"]
        return data


class BaseDynamicPullCorrector(BasePullCorrector):

    def pull_correct_static(self, data):
        data = self.floor.apply_dynamic(data)
        data = self.pull_correct(data)
        return data


class BaseMedianPullCorrector(BasePullCorrector):

    def __init__(self, pull_dict):
        BasePullCorrector.__init__(self, pull_dict)

        if not os.path.isfile(self.pull_name):
            self.create_pickle()
        else:
            print "Loading from", self.pull_name

        self.create_pickle()

        with open(self.pull_name, "r") as f:
            pickled_data = Pickle.load(f)

        self.f = self.create_function(pickled_data)

    def create_pickle(self):
        pass

    def create_function(self, pickled_array):
        pass


@BasePullCorrector.register_subclass("median_0d_e")
class MedianPullEParam0D(BaseMedianPullCorrector, BaseDynamicFloorClass):

    def create_pickle(self):
        create_pull_0d_e(self.pull_dict)

# @BasePullCorrector.register_subclass('static_pull_corrector')
# class Static1DPullCorrector(BasePullCorrector):
#
#     def __init__(self, season, e_pdf_dict, **kwargs):
#         BasePullCorrector.__init__(self, season, e_pdf_dict)


from flarestack.shared import fs_scratch_dir
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
import json

path = fs_scratch_dir + "tester_spline.npy"

e_pdf_dict = {
    "Name": "Power Law",
    "Gamma": 3.0
}

x = BasePullCorrector.create(IC86_1_dict, e_pdf_dict, "quantile_floor_1d_e",
                             "median_0d_e")

# x = BaseFloorClass.create(IC86_1_dict, e_pdf_dict, "quantile_floor_1d_e",
#                           floor_quantile=0.1)
# for gamma in np.linspace(1.0, 4.0, 4):
#     print data_loader(IC86_1_dict["exp_path"])["logE"][:8]
#     print np.degrees(x.f(data_loader(IC86_1_dict["exp_path"])[:8], [gamma]))

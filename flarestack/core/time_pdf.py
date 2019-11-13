import logging
import numpy as np
from scipy.interpolate import interp1d
from flarestack.icecube_utils.dataset_loader import data_loader


def box_func(t, t0, t1):
    """Box function that is equal to 1 between t0 and t1, and 0 otherwise.
    Equal to 0.5 at to and t1.

    :param t: Time to be evaluated
    :param t0: Start time of box
    :param t1: End time of box
    :return: Value of Box function at t
    """
    val = 0.5 * ((np.sign(t - t0)) - (np.sign(t - t1)))
    return val


def read_t_pdf_dict(t_pdf_dict):
    """Ensures backwards compatibility for t_pdf_dict objects"""

    maps = [
        ("Offset", "offset"),
        ("Pre-Window", "pre_window"),
        ("Post-Window", "post_window"),
        ("Fixed Ref Time (MJD)", "fixed_ref_time_mjd"),
        ("Name", "time_pdf_name"),
        ("Max Offset", "max_offset"),
        ("Min Offset", "min_offset"),
        ("Max Flare", "max_flare")
    ]

    for (old_key, new_key) in maps:

        if old_key in list(t_pdf_dict.keys()):
            logging.warning("Deprecated t_pdf_key '{0}' was used. "
                            "Please use '{1}' in future.".format(old_key, new_key))
            t_pdf_dict[new_key] = t_pdf_dict[old_key]

    name_maps = [
        ("Steady", "steady"),
        ("Box", "box"),
        ("FixedRefBox", "fixed_ref_box"),
        ("FixedEndBox", "custom_source_box") # Not a typo! Class renamed.
    ]

    for (old_key, new_key) in name_maps:
        if t_pdf_dict["time_pdf_name"] == old_key:
            logging.warning("Deprecated time_pdf_name '{0}' was used. "
                            "Please use '{1}' in future.".format(old_key, new_key))
            t_pdf_dict["time_pdf_name"] = new_key

    return t_pdf_dict


class TimePDF(object):
    subclasses = {}

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        self.t_dict = t_pdf_dict

        if livetime_pdf is not None:
            self.livetime_f = lambda x: livetime_pdf.livetime_f(x)# * livetime_pdf.livetime
            self.livetime_pdf = livetime_pdf
            self.t0 = livetime_pdf.sig_t0()
            self.t1 = livetime_pdf.sig_t1()
            self.mjd_to_livetime, self.livetime_to_mjd = self.livetime_pdf.get_mjd_conversion()

        else:
            self.livetime_pdf = None
            self.t0 = -np.inf
            self.t1 = np.inf
            self.livetime = None

    @classmethod
    def register_subclass(cls, time_pdf_name):
        def decorator(subclass):
            cls.subclasses[time_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, t_pdf_dict, livetime_pdf=None):

        t_pdf_dict = read_t_pdf_dict(t_pdf_dict)

        t_pdf_name = t_pdf_dict["time_pdf_name"]

        if t_pdf_name not in cls.subclasses:
            raise ValueError('Bad time PDF name {}'.format(t_pdf_name))

        return cls.subclasses[t_pdf_name](t_pdf_dict, livetime_pdf)

    def product_integral(self, t, source):
        """Calculates the product of the given signal PDF with the season box
        function. Thus gives 0 everywhere outside the season, and otherwise
        the value of the normalised integral. The season function is offset
        by 1e-9, to ensure that f(t1) is equal to 1. (i.e the function is
        equal to 1 at the end of the box).

        :param t: Time
        :param source: Source to be considered
        :return: Product of signal integral and season
        """

        f = np.array(self.signal_integral(t, source))

        f[f < 0.] = 0.
        f[f > 1.] = 1.

        return f

    def inverse_interpolate(self, source):
        """Calculates the values for the integral of the signal PDF within
        the season. Then rescales these values, such that the start of the
        season yields 0, and then end of the season yields 1. Creates a
        function to interpolate between these values. Then, for a number
        between 0 and 1, the interpolated function will return the MJD time
        at which that fraction of the cumulative distribution was reached.

        :param source: Source to be considered
        :return: Interpolated function
        """
        max_int = self.product_integral(self.sig_t1(source), source)
        min_int = self.product_integral(self.sig_t0(source), source)
        fraction = max_int - min_int

        t_range = np.linspace(self.sig_t0(source), self.sig_t1(source),
                              int(1e4))
        cumu = (self.product_integral(t_range, source) - min_int) / fraction

        # Checks to ensure the cumulative fraction spans 0 to 1
        if max(cumu) > 1:
            raise Exception("Cumulative Distribution exceeds 1.")
        elif min(cumu) < 0:
            raise Exception("Cumulative Distribution extends below 0.")

        return interp1d(cumu, t_range, kind='linear')

    def simulate_times(self, source, n_s):
        """Randomly draws times for n_s events for a given source,
        all lying within the current season. The values are based on an
        interpolation of the integrated time PDF.

        :param source: Source being considered
        :param n_s: Number of event times to be simulated
        :return: Array of times in MJD for a given source
        """
        f = self.inverse_interpolate(source)

        sims = f(np.random.uniform(0., 1., n_s))

        return sims

    def f(self, t, source):
        raise NotImplementedError(
            "No 'f' has been implemented for {0}".format(
                self.__class__.__name__
            ))

    def sig_t0(self, source):
        """Calculates the starting time for the time pdf.

        :param source: Source to be considered
        :return: Time of PDF start
        """
        raise NotImplementedError("sig_t0 function not implemented for "
                                  "{0}".format(self.__class__.__name__))

    def sig_t1(self, source):
        """Calculates the ending time for the time pdf.

        :param source: Source to be considered
        :return: Time of PDF end
        """
        raise NotImplementedError("sig_t1 function not implemented for "
                                  "{0}".format(self.__class__.__name__))

    def integral_to_infinity(self, source):
        max_int = self.product_integral(self.sig_t1(source), source)
        min_int = self.product_integral(self.sig_t0(source), source)
        return max_int - min_int

    # def convert_livetime_mjd(self):
    #     t_range = np.linspace(
    #         self.livetime_pdf.sig_t0(), self.livetime_pdf.sig_t1(), int(1e3))
    #
    #     f_range = np.array([self.livetime_pdf.livetime_f(t) for t in t_range])
    #     f_range /= np.sum(f_range)
    #
    #     sum_range = [np.sum(f_range[:i]) for i, _ in enumerate(f_range)]
    #
    #     mjd_to_livetime = interp1d(t_range, sum_range)
    #     livetime_to_mjd = interp1d(sum_range, t_range)
    #     return mjd_to_livetime, livetime_to_mjd

    def effective_injection_time(self, source=None):
        raise NotImplementedError

    def get_livetime(self):
        raise NotImplementedError

    def get_mjd_conversion(self):
        raise NotImplementedError

@TimePDF.register_subclass('steady')
class Steady(TimePDF):
    """The time-independent case for a Time PDF. Requires no additional
    arguments in the dictionary for __init__. Used for a steady source that
    is continuously emitting.
    """

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        TimePDF.__init__(self, t_pdf_dict, livetime_pdf)

        if self.livetime_pdf is None:
            raise ValueError("No livetime pdf has been provided, but a Steady "
                             "Time PDF has been chosen. Without a fixed start "
                             "and end point, no PDF can be defined. Please "
                             "provide a livetime_pdf, or use a different Time "
                             "PDF class such as FixedEndBox.")
        else:
            self.livetime = livetime_pdf.get_livetime()

    def f(self, t, source):
        """In the case of a steady source, the signal PDF is a uniform PDF in
        time. It is thus simply equal to the season_f, normalised with the
        length of the season to give an integral of 1. It is thus equal to
        the background PDF.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        return 1./self.livetime#self.livetime_f(t)# * self.livetime

    def signal_integral(self, t, source):
        """In the case of a steady source, the signal PDF is a uniform PDF in
        time. Thus, the integral is simply a linear function increasing
        between t0 (box start) and t1 (box end). After t1, the integral is
        equal to 1, while it is equal to 0 for t < t0.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """

        return self.mjd_to_livetime(t) / self.livetime

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        entire season. Thus returns the start and end times for the season.

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = self.t0
        end = self.t1

        return start, end

    def effective_injection_time(self, source=None):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        season_length = self.integral_to_infinity(source) * self.livetime

        return season_length * (60 * 60 * 24)

    def raw_injection_time(self, source):
        """Calculates the 'raw injection time' which is the injection time
        assuming a detector with 100% uptime. Useful for calculating source
        emission times for source-frame energy estimation.

        :param source: Source to be considered
        :return: Time in seconds for 100% uptime
        """
        return (self.t1 - self.t0) * (60 * 60 * 24)

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """
        return self.t0

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return self.t1


@TimePDF.register_subclass('box')
class Box(TimePDF):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. Requires arguments of
    Pre-Window and Post_window, and gives a box from Pre-Window days before
    the reference time to Post-Window days after the reference time.
    """

    def __init__(self, t_pdf_dict, season):

        TimePDF.__init__(self, t_pdf_dict, season)
        self.pre_window = self.t_dict["pre_window"]
        self.post_window = self.t_dict["post_window"]

        if "offset" in list(t_pdf_dict.keys()):
            self.offset = self.t_dict["offset"]
            self.pre_window -= self.offset
            self.post_window += self.offset

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """
        return max(self.t0, source["ref_time_mjd"] - self.pre_window)

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return min(source["ref_time_mjd"] + self.post_window, self.t1)

    def f(self, t, source=None):
        """In this case, the signal PDF is a uniform PDF for a fixed duration of
        time. It is normalised with the length of the box in LIVETIME rather
        than days, to give an integral of 1.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        length = self.mjd_to_livetime(t1) - self.mjd_to_livetime(t0)

        if length > 0.:
            return box_func(t, t0, t1) / length

        else:
            return np.zeros_like(t)

    def signal_integral(self, t, source):
        """In this case, the signal PDF is a uniform PDF for a fixed duration of
        time. Thus, the integral is simply a linear function increasing
        between t0 (box start) and t1 (box end). After t1, the integral is
        equal to 1, while it is equal to 0 for t < t0.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """

        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        length = self.mjd_to_livetime(t1) - self.mjd_to_livetime(t0)

        return np.abs((self.mjd_to_livetime(t) - self.mjd_to_livetime(t0))
                      * box_func(t, t0, t1+1e-9) / length)

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        period of overlap of the flare and the box. Thus, for a given season,
        return the source and data

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = max(self.t0, self.sig_t0(source))
        end = min(self.t1, self.sig_t1(source))

        return start, end

    def effective_injection_time(self, source=None):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        t0 = self.mjd_to_livetime(self.sig_t0(source))
        t1 = self.mjd_to_livetime(self.sig_t1(source))
        time = (t1 - t0) * 60 * 60 * 24

        return max(time, 0.)

    def raw_injection_time(self, source):
        """Calculates the 'raw injection time' which is the injection time
        assuming a detector with 100% uptime. Useful for calculating source
        emission times for source-frame energy estimation.

        :param source: Source to be considered
        :return: Time in seconds for 100% uptime
        """

        diff = max(self.sig_t1(source) - self.sig_t0(source), 0)
        return diff * (60 * 60 * 24)

@TimePDF.register_subclass('fixed_ref_box')
class FixedRefBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is unique for each source. The sources must have
    a field "Start Time (MJD)" and another "End Time (MJD)", specifying the
    period of the Time PDF.
    """
    def __init__(self, t_pdf_dict, season):
        Box.__init__(self, t_pdf_dict, season)
        self.fixed_ref = t_pdf_dict["fixed_ref_time_mjd"]

    def sig_t0(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        return max(self.t0, self.fixed_ref - self.pre_window)

    def sig_t1(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return min(self.fixed_ref + self.post_window, self.t1)


@TimePDF.register_subclass('fixed_end_box')
class FixedEndBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is the same for all sources.
    """

    def __init__(self, t_pdf_dict, season):
        self.start_time_mjd = t_pdf_dict["start_time_mjd"]
        self.end_time_mjd = t_pdf_dict["end_time_mjd"]
        if "offset" in t_pdf_dict:
            self.offset = self.t_dict["offset"]
        else:
            self.offset = 0
        TimePDF.__init__(self, t_pdf_dict, season)


    def sig_t0(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        t0 = self.start_time_mjd + self.offset

        return max(self.t0, t0)

    def sig_t1(self, source=None):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """

        t1 = self.end_time_mjd + self.offset

        return min(t1, self.t1)

    def get_livetime(self):
        if self.livetime_pdf is not None:
            raise Exception("Livetime PDF already provided.")
        else:
            return self.sig_t1() - self.sig_t0()

    def get_mjd_conversion(self):
        if self.livetime_pdf is not None:
            raise Exception("Livetime PDF already provided.")
        else:
            mjd_to_l = lambda x: x - self.sig_t0()
            l_to_mjd = lambda x: x + self.sig_t0()
            return mjd_to_l, l_to_mjd

@TimePDF.register_subclass('custom_source_box')
class CustomSourceBox(Box):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. In this case, the start
    and end time for the box is unique for each source. The sources must have
    a field "Start Time (MJD)" and another "End Time (MJD)", specifying the
    period of the Time PDF.
    """

    def __init__(self, t_pdf_dict, season):
        TimePDF.__init__(self, t_pdf_dict, season)
        if "offset" in t_pdf_dict:
            self.offset = self.t_dict["offset"]
        else:
            self.offset = 0

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        t0 = source["start_time_mjd"] + self.offset

        return max(self.t0, t0)

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """

        t1 = source["end_time_mjd"] + self.offset

        return min(t1, self.t1)


@TimePDF.register_subclass("detector_on_off_list")
class DetectorOnOffList(TimePDF):
    """TimePDF with predefined on/off periods. Can be used for a livetime
    function, in which observations are divided into runs with gaps. Can also
    be used for e.g pre-defined interesting period for a variable source.
    """

    def __init__(self, t_pdf_dict, livetime_pdf=None):
        TimePDF.__init__(self, t_pdf_dict, livetime_pdf)

        try:
            self.on_off_list = t_pdf_dict["on_off_list"]
        except KeyError:
            raise KeyError("No 'on_off_list' found in t_pdf_dict. The "
                           "following fields were provided: {0}".format(
                t_pdf_dict.keys()
            ))
        self.t0, self.t1, self._livetime, self.season_f, self.mjd_to_livetime, self.livetime_to_mjd = self.parse_list()

    def parse_list(self):
        t0 = min(self.on_off_list["start"])
        t1 = max(self.on_off_list["stop"])
        livetime = np.sum(self.on_off_list["length"])
        return t0, t1, livetime

    def f(self, t, source=None):
        return self.season_f(t)/self.livetime

    def livetime_f(self, t, source=None):
        return self.f(t, source)

    def sig_t0(self, source=None):
        return self.t0

    def sig_t1(self, source=None):
        return self.t1

    def parse_list(self):
        raise NotImplementedError

    def get_livetime(self):
        return self._livetime

    def get_mjd_conversion(self):
        return self.mjd_to_livetime, self.livetime_to_mjd



# from data.icecube_pointsource_7_year import ps_v002_p01
# from shared import catalogue_dir
#
# cat_path = catalogue_dir + "TDEs/individual_TDEs/Swift J1644+57_catalogue.npy"
# catalogue = np.load(cat_path)
#
# for t in np.linspace(0, 500, 101):
#     time_dict = {
#         "Name": "FixedRefBox",
#         "Fixed Ref Time (MJD)": 55650,
#         "Pre-Window": 0,
#         "Post-Window": t
#     }
#
#     livetime = 0
#     tot = 0
#
#     for season in ps_v002_p01[-2:-1]:
#         tpdf = TimePDF.create(time_dict, season)
#         livetime += tpdf.effective_injection_time(
#             catalogue
#         )
#         tot += tpdf.livetime
#
#     print t, livetime/(60*60*24), tot

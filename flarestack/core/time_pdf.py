from __future__ import print_function
from __future__ import division
from builtins import input
from builtins import object
import numpy as np
from scipy.interpolate import interp1d
from flarestack.utils.dataset_loader import grl_loader, data_loader


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
            t_pdf_dict[new_key] = t_pdf_dict[old_key]

    return t_pdf_dict


class TimePDF(object):
    subclasses = {}

    def __init__(self, t_pdf_dict, season):
        self.t_dict = t_pdf_dict
        self.season = season

        if season["grl_path"] is not None:

            self.grl = grl_loader(season)

            self.t0 = min(self.grl["start"])
            self.t1 = max(self.grl["stop"])

            self.livetime = np.sum(self.grl["length"])

            step = 1e-10

            t_range = [self.t0 - step]
            f = [0.]

            mjd = [0.]
            livetime = [0.]
            total_t = 0.

            for i, run in enumerate(self.grl):

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

            stitch_t = [t_range[0]]
            stitch_f = [1.]
            for i, t in enumerate(t_range[1:]):
                gap = t - t_range[i - 1]

                if gap < 1e-5 and f[i] == 0:
                    pass
                else:
                    stitch_t.append(t)
                    stitch_f.append(f[i])

            if stitch_t != sorted(stitch_t):
                print("Error in ordering GoodRunList!")
                print("Runs are out of order!")

                print(self.grl[:5])
                input("prompt")

                for j, t in enumerate(stitch_t):
                    if t != sorted(stitch_t)[j]:
                        print(j, t, self.grl[j])
                input("prompt")

            mjd.append(1e5)
            livetime.append(total_t)

            self.season_f = interp1d(stitch_t, stitch_f, kind="linear")

            self.mjd_to_livetime = interp1d(mjd, livetime, kind="linear")
            self.livetime_to_mjd = interp1d(livetime, mjd, kind="linear")

        else:
            exp = data_loader(season["exp_path"])
            self.t0 = min(exp["time"])
            self.t1 = max(exp["time"])
            self.livetime = self.t1 - self.t0
            self.season_f = lambda x: 1.
            self.mjd_to_livetime = lambda x: x - self.t0
            self.livetime_to_mjd = lambda x: x + self.t0

    @classmethod
    def register_subclass(cls, time_pdf_name):
        def decorator(subclass):
            cls.subclasses[time_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, t_pdf_dict, season):

        t_pdf_dict = read_t_pdf_dict(t_pdf_dict)

        t_pdf_name = t_pdf_dict["time_pdf_name"]

        if t_pdf_name not in cls.subclasses:
            raise ValueError('Bad time PDF name {}'.format(t_pdf_name))

        return cls.subclasses[t_pdf_name](t_pdf_dict, season)

    def background_f(self, t, source):
        """In all cases, we assume that the background is uniform in time.
        Thus, the background PDF is just a normalised version of the season_f
        box function.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        return 1. / self.livetime

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

        t_range = np.linspace(self.sig_t0(source), self.sig_t1(source), 1e4)
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

        return f(np.random.uniform(0., 1., n_s))


@TimePDF.register_subclass('Steady')
class Steady(TimePDF):
    """The time-independent case for a Time PDF. Requires no additional
    arguments in the dictionary for __init__. Used for a steady source that
    is continuously emitting.
    """

    def signal_f(self, t, source):
        """In the case of a steady source, the signal PDF is a uniform PDF in
        time. It is thus simply equal to the season_f, normalised with the
        length of the season to give an integral of 1. It is thus equal to
        the background PDF.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        return self.background_f(t, source)

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

    def effective_injection_time(self, source):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        season_length = self.livetime

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


@TimePDF.register_subclass('Box')
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

    def signal_f(self, t, source):
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

    def effective_injection_time(self, source):
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


@TimePDF.register_subclass('FixedRefBox')
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

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """

        return max(self.t0, self.fixed_ref - self.pre_window)

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return min(self.fixed_ref + self.post_window, self.t1)


@TimePDF.register_subclass('FixedEndBox')
class FixedEndBox(Box):
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


# from data.icecube_pointsource_7_year import ps_7year
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
#     for season in ps_7year[-2:-1]:
#         tpdf = TimePDF.create(time_dict, season)
#         livetime += tpdf.effective_injection_time(
#             catalogue
#         )
#         tot += tpdf.livetime
#
#     print t, livetime/(60*60*24), tot

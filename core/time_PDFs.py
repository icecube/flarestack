import numpy as np
from scipy.interpolate import interp1d


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


class TimePDF:
    subclasses = {}

    def __init__(self, t_pdf_dict, season):
        self.t_dict = t_pdf_dict
        self.season = season
        self.t0 = season["Start (MJD)"]
        self.t1 = season["End (MJD)"]
        self.livetime = season["Livetime"]
        self.season_f = lambda t: box_func(t, self.t0 - 1e-9, self.t1 + 1e-9)

    @classmethod
    def register_subclass(cls, time_pdf_name):
        def decorator(subclass):
            cls.subclasses[time_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, t_pdf_dict, season):

        t_pdf_name = t_pdf_dict["Name"]

        if t_pdf_name not in cls.subclasses:
            raise ValueError('Bad time PDF name {}'.format(t_pdf_name))

        return cls.subclasses[t_pdf_name](t_pdf_dict, season)

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
        return self.signal_integral(t, source) * self.season_f(t)

    def effective_injection_time(self, source):
        """Calculates the effective injection time for the given PDF.
        The livetime is measured in days, but here is converted to seconds.

        :param source: Source to be considered
        :return: Effective Livetime in seconds
        """
        season_length = self.season["Livetime"]
        frac = self.product_integral(self.t1, source) - \
               self.product_integral(self.t0, source)

        return season_length * (60 * 60 * 24) * frac

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
        max_int = self.product_integral(self.t1, source)
        min_int = self.product_integral(self.t0, source)
        fraction = max_int - min_int

        t_range = np.linspace(self.t0, self.t1, 1.e4)
        cumu = (self.product_integral(t_range, source) - min_int) / fraction

        # Checks to ensure the cuumulative fraction spans 0 to 1
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

    def background_f(self, t, source):
        """In all cases, we assume that the background is uniform in time.
        Thus, the background PDF is just a normalised version of the season_f
        box function.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """
        return self.season_f(t) / (self.t1 - self.t0)

    def time_weight(self, source):
        diff = self.signal_integral(self.t1, source) - \
               self.signal_integral(self.t0, source)

        time = self.t1 - self.t0
        return diff * time


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

        return np.abs(((t - self.t0) * (self.signal_f(t, source)) +
                       0.5 * (np.sign(t - self.t1) + 1)))

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        entire season. Thus returns the start and end times for the season.

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = self.t0
        end = self.t1

        return start, end


@TimePDF.register_subclass('Box')
class Box(TimePDF):
    """The simplest time-dependent case for a Time PDF. Used for a source that
    is uniformly emitting for a fixed period of time. Requires arguments of
    Pre-Window and Post_window, and gives a box from Pre-Window days before
    the reference time to Post-Window days after the reference time.
    """

    def __init__(self, t_pdf_dict, season):
        TimePDF.__init__(self, t_pdf_dict, season)
        self.pre_window = self.t_dict["Pre-Window"]
        self.post_window = self.t_dict["Post-Window"]

    def sig_t0(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD minus the length of the pre-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window Start
        """
        return source["Ref Time (MJD)"] - self.pre_window

    def sig_t1(self, source):
        """Calculates the starting time for the window, equal to the
        source reference time in MJD plus the length of the post-reference-time
        window (in days).

        :param source: Source to be considered
        :return: Time of Window End
        """
        return source["Ref Time (MJD)"] + self.post_window

    def signal_f(self, t, source):
        """In this case, the signal PDF is a uniform PDF for a fixed duration of
        time. It is normalised with the length of the box, to give an
        integral of 1.

        :param t: Time
        :param source: Source to be considered
        :return: Value of normalised box function at t
        """

        t0 = self.sig_t0(source)
        t1 = self.sig_t1(source)

        return box_func(t, t0, t1) / (t1 - t0)

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

        return np.abs(((t - t0) * (self.signal_f(t, source)) +
                       0.5 * (np.sign(t - t1) + 1)))

    def flare_time_mask(self, source):
        """In this case, the interesting period for Flare Searches is the
        period of overlap of the flare and the box. Thus, for a given season,
        return the source and data

        :return: Start time (MJD) and End Time (MJD) for flare search period
        """

        start = max(self.t0, self.sig_t0(source))
        end = min(self.t1, self.sig_t1(source))

        return start, end
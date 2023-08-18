import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import scipy.optimize, scipy.stats
from scipy.stats import norm


logger = logging.getLogger(__name__)

n_bins = 100


def get_ts_fit_type(mh_dict):
    """Function to select the best method for fitting the TS distribution resulting
    from a particular minimisation_handler.

    :param mh_dict: MinimisationHandler or ResultsHandler dictionary
    :return: name for ts_type
    """
    if mh_dict["mh_name"] == "flare":
        ts_fit_type = "flare"
    else:
        ts_fit_type = "standard"

    return ts_fit_type


# Taken via Alex Stasik from Thomas Kintscher


class Chi2_LeftTruncated(object):
    """A class similar to the ones from scipy.stats
    allowing to fit left-truncated chi^2 distributions.
    """

    def __init__(self, data, cut=0.0, **kwargs):
        """Fit the given ensemble of measurements with a chi^2 function.

        `data` is a list of test statistics values.
        `cut` defines where the distribution is truncated.
        """

        data_left = data[data <= cut]
        data_right = data[data > cut]
        N_all = len(data)
        N_left = len(data_left)

        # three parameters will be fitted: dof, location, scale
        p_start = [2.0, -1.0, 1.0]
        p_bounds = [
            (0.0, None),  # dof > 0
            (None, -0),  # location < 0 for 'truncated' effect
            (1e-5, 1e5),
        ]  # shape ~ free

        # define the fit function: likelihood for chi^2 distribution,
        # plus knowledge about the amount of truncated data
        def func(p):
            dist = scipy.stats.chi2(*p)
            loglh = dist.logpdf(data_right).sum()
            loglh += N_left * dist.logcdf(cut)

            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        # self._q_left = N_left / float(N_all)
        self._cut = cut
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data_right, self._f.cdf)[0]
        self.ndof = res.x[0]
        self.loc = res.x[1]
        self.scale = res.x[2]
        self._res = res

    def pdf(self, x):
        """Probability density function."""
        m_flat = (0.0 <= x) & (x <= self._cut)
        m_chi2 = x > self._cut
        x = np.asarray(x, dtype=object)
        r = np.zeros_like(x)
        if self._cut == 0.0:
            r[m_flat] = self._f.cdf(self._cut)
        else:
            r[m_flat] = self._f.cdf(self._cut) / self._cut
        r[m_chi2] = self._f.pdf(x[m_chi2])
        return r

    def cdf(self, x):
        """Cumulative distribution function."""
        return self._f.cdf(x)

    def sf(self, x):
        """Survival function."""
        return self._f.sf(x)

    def isf(self, x):
        """Inverse survival function."""
        return self._f.isf(x)

    def ppf(self, x):
        """Percent point function."""
        return self._f.ppf(x)

    def __str__(self):
        return (
            "Left-truncated Chi^2 Distribution:\n"
            + "\t DoF      = {0:7.2f}\n".format(self._f.args[0])
            + "\t Location = {0:7.2f}\n".format(self._f.args[1])
            + "\t Scale    = {0:7.2f}\n".format(self._f.args[2])
            + "\t KS       = {0:7.2%}".format(self._ks)
        )


class Chi2_one_side(object):
    def __init__(self, data):
        p_start = [2.0, -1.0, 1.0]
        p_start = [1.3]
        p_bounds = [
            (0.0, None),  # dof > 0
        ]
        # (1e-5, 1e5)]  # shape ~ free

        def func(p):
            loglh = scipy.stats.chi2(p[0], loc=0.0, scale=1.0).logpdf(data).sum()
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)
        hess_inv = res.hess_inv.todense()
        sigma = np.sqrt(hess_inv)

        self._cut = 0.0
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self._res = res
        self.sigma = sigma


class Chi2_one_side_free(object):
    def __init__(self, data):
        # p_start = [4., -1., 1.]
        p_start = [2.0, -1.0, 1.0]
        p_bounds = [
            (0.0, None),  # dof > 0
            (None, -0),  # location < 0 for 'truncated' effect
            (1e-5, 1e5),  # shape ~ free
        ]
        # (1e-5, 1e5)]  # shape ~ free

        def func(p):
            loglh = (
                scipy.stats.chi2(p[0], loc=p[1] - 1e-8, scale=p[2]).logpdf(data).sum()
            )
            # print loglh, p
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        # print res

        self._cut = 0.0
        self._res = res
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self.ndof = res.x[0]
        self.loc = res.x[1]
        self.scale = res.x[2]


def fit_background_ts(ts_array: np.array, ts_type):
    """
    Fit the background TS distribution
    """
    mask = ts_array > 0.0

    frac_positive = float(len(ts_array[mask])) / (float(len(ts_array)))

    threshold_err = 0.0

    ts_data, labels, colors = [], [], []

    if np.sum(mask) > 0.0:
        ts_data.append(ts_array[mask])
        labels.append("TS > 0")
        colors.append("black")
    if np.sum(~mask) > 0.0:
        ts_data.append(ts_array[~mask])
        labels.append("TS <= 0")
        colors.append("grey")

    plt.hist(
        ts_data,
        bins=n_bins,
        lw=2,
        histtype="step",
        color=colors,
        label=labels,
        density=True,
        stacked=True,
    )

    if ts_type == "flare":
        chi2 = Chi2_LeftTruncated(ts_array)

        if chi2._res.success:
            frac_positive = 1.0

            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

        else:
            chi2 = Chi2_one_side_free(ts_array[ts_array > 0.0])
            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

            if not chi2._res.success:
                chi2 = Chi2_one_side(ts_array[ts_array > 0.0])
                df = chi2._f.args[0]
                loc = 0.0
                scale = 1.0

    elif ts_type == "fit_weight":
        chi2 = Chi2_one_side_free(ts_array[ts_array > 0.0])

        if chi2._res.success:
            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

        else:
            chi2 = Chi2_LeftTruncated(ts_array)
            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

            if not chi2._res.success:
                chi2 = Chi2_one_side(ts_array[ts_array > 0.0])
                df = chi2._f.args[0]
                loc = 0.0
                scale = 1.0

            else:
                frac_positive = 1.0

    elif ts_type in ["standard", "negative_ns"]:
        chi2 = Chi2_one_side(ts_array[ts_array > 0.0])

        df = chi2._f.args[0]
        loc = 0.0
        scale = 1.0
        threshold_err = float(chi2.sigma)

    else:
        raise Exception(f"ts_type {ts_type} not recognised!")

    return df, loc, scale, frac_positive, threshold_err


def plot_expanded_negative(ts_array, path):
    plt.figure()
    plt.hist(
        [ts_array[ts_array > 0], ts_array[ts_array < 0]],
        bins=n_bins,
        lw=2,
        histtype="step",
        color=["black", "grey"],
        label=["TS > 0", "TS <= 0"],
        density=True,
        stacked=True,
    )

    med = np.median(ts_array)

    plt.yscale("log")
    plt.xlabel(r"Test statistic ($\lambda$)")
    plt.legend(loc="upper right")
    plt.savefig(path[:-4] + "_expanded.pdf")
    plt.close()


def filter_nan(values: list) -> np.array:
    arr = np.array(values)

    nan_count = np.sum(np.isnan(arr))

    if nan_count > 0:
        logger.warning(f"TS distribution has {nan_count} NaN entries.")

    return arr[~np.isnan(arr)]


def calc_ts_threshold(ts_values: list, significance: float, ts_type: str = "standard"):
    ts_arr = filter_nan(ts_values)


def plot_background_ts_distribution(
    ts_values: list,
    path: Path,
    ts_type: str = "standard",
    significance_threshold: float = 5.0,
    ts_val=None,
    mock_unblind: bool = False,
):
    path.mkdir(exist_ok=True)

    # find max ts
    ts_arr = filter_nan(ts_values)
    max_ts = np.max(ts_arr)

    if np.median(ts_arr) < 0.0:
        # plot separately the positive and negative values of TS
        plot_expanded_negative(ts_arr, path)

    if max_ts == 0:
        logger.warning(
            f"Maximum of TS is zero, unable to calculate the TS threshold for the required significance (too few trials?)"
        )
        return np.NaN

    df, loc, scale, frac_positive, t_err = fit_background_ts(ts_arr, ts_type)

    frac_nonpositive = 1.0 - frac_positive

    # cumulative fraction of trials required to reach the desired significance
    cdf_threshold = norm.cdf(significance_threshold)
    # determine corresponding threshold for the CDF of
    positive_cdf_threshold = (cdf_threshold - frac_nonpositive) / frac_positive

    disc_potential = scipy.stats.chi2.ppf(positive_cdf_threshold, df, loc, scale)

    # plotting
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    plt.axhline(frac_positive * (1 - positive_cdf_threshold), color="r", linestyle="--")

    x_range = np.linspace(0.0, max(max_ts, disc_potential), 100)

    plt.plot(
        x_range,
        frac_positive * scipy.stats.chi2.pdf(x_range, df, loc, scale),
        color="blue",
        label=r"$\chi^{2}$ distribution",
    )

    if t_err is not None:
        plt.fill_between(
            x_range,
            frac_positive * scipy.stats.chi2.pdf(x_range, df + t_err, loc, scale),
            frac_positive * scipy.stats.chi2.pdf(x_range, df - t_err, loc, scale),
            alpha=0.1,
            color="blue",
        )

    def integral(x):
        return frac_nonpositive * np.sign(x) + frac_positive * (
            scipy.stats.chi2.cdf(x, df, loc, scale)
        )

    plt.plot(
        x_range,
        1.0 - integral(x_range),
        color="green",
        linestyle="--",
        label=r"1 - $\int f(x)$ (p-value)",
    )

    plt.axvline(
        disc_potential, color="r", label=rf"{significance_threshold} $\sigma$ threshold"
    )

    if ts_val is not None:
        if not isinstance(ts_val, float):
            ts_val = float(ts_val[0])

        logger.info(f"Quantifying TS: {ts_val:.2f}")

        if ts_val > np.median(ts_arr):
            # val = (ts_val - frac_under) / (1. - frac_under)

            cdf = frac_nonpositive + frac_positive * scipy.stats.chi2.cdf(
                ts_val, df, loc, scale
            )

            sig = norm.ppf(cdf)

        else:
            cdf = 0.0
            sig = 0.0

        logger.info(f"Pre-trial P-value is {1-cdf:.2E}")
        logger.info(f"Significance is {sig:.2f} sigma")

        plt.axvline(
            ts_val,
            color="purple",
            label="{:.2f}".format(ts_val)
            + " TS/"
            + "{:.2f}".format(sig)
            + r" $\sigma$",
        )

    else:
        plt.annotate(
            "{:.1f}".format(100 * frac_nonpositive)
            + "% of data in delta. \n"
            + r"$\chi^{2}$ Distribution:"
            + "\n   * d.o.f.="
            + "{:.2f} \pm {:.2f}".format(df, t_err)
            + ",\n  * loc="
            + "{:.2f}".format(loc)
            + " \n * scale="
            + "{:.2f}".format(scale),
            xy=(0.1, 0.2),
            xycoords="axes fraction",
            fontsize=8,
        )

    yrange = min(
        1.0 / (float(len(ts_arr)) * n_bins),
        scipy.stats.chi2.pdf(disc_potential, df, loc, scale),
    )

    plt.yscale("log")
    plt.xlabel(r"Test statistic ($\lambda$)")
    plt.legend(loc="upper right")

    if mock_unblind:
        ax.text(0.2, 0.5, "MOCK DATA", color="grey", alpha=0.5, transform=ax.transAxes)

    plt.savefig(path)
    plt.close()

    return disc_potential


def plot_fit_results(results, path, inj):
    # results = np.array(results)

    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

    n_dim = len(list(results.keys()))

    try:
        fig = plt.figure()

        fig.set_size_inches(7, n_dim * 3)
        fig.subplots_adjust(hspace=0.5)

        for i, (label, row) in enumerate(results.items()):
            weights = np.ones(len(row)) / float(len(row))

            plt.subplot(n_dim, 1, i + 1)
            plt.hist(row, histtype="step", weights=weights, bins=100, color="blue")
            plt.axvline(np.median(row), linestyle="--", color="blue", label="Median")
            plt.title(label)

            plt.axvline(inj[label], linestyle="--", color="orange", label="Injection")

            plt.legend()

        plt.savefig(path)
        plt.close()

    except ValueError:
        pass

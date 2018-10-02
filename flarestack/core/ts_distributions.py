import numpy as np
import os
from flarestack.shared import plots_dir
import matplotlib.pyplot as plt
import scipy.optimize, scipy.stats
from scipy.stats import norm

raw_five_sigma = norm.cdf(5)
n_bins = 100

class Chi2:

    """ A class similar to the ones from scipy.stats
       allowing to fit left-truncated chi^2 distributions.
    """

    def __init__(self, data):
        """ Fit the given ensemble of measurements with a chi^2 function.

        `data` is a list of test statistics values.
        `cut` defines where the distribution is truncated.
        """

        data -= min(data)

        # three parameters will be fitted: dof, location, scale
        p_start = [2., 1.]
        p_bounds = [(0., None),  # dof > 0
                    # (min(data), 0.),  # location < 0 for 'truncated'
                    # effect
                    (1e-5, 1e5)]  # shape ~ free

        # define the fit function: likelihood for chi^2 distribution,
        # plus knowledge about the amount of truncated data
        def func(p):

            dist = scipy.stats.chi2(p[0], loc=0., scale=p[1])
            loglh = dist.logpdf(data).sum()
            print loglh

            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        if not res.success:
            print 'Chi2 fit did not converge! Result is likely garbage.'

        # self._q_left = N_left / float(N_all)
        self._cut = 0.
        self._f = scipy.stats.chi2(res.x[0], loc=min(data), scale=res.x[1])
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self.ndof = res.x[0]
        self.loc = min(data)
        self.scale = res.x[1]



class Chi2_LeftTruncated(object):
    """ A class similar to the ones from scipy.stats
       allowing to fit left-truncated chi^2 distributions.
    """

    def __init__(self, data, cut=0., **kwargs):
        """ Fit the given ensemble of measurements with a chi^2 function.

        `data` is a list of test statistics values.
        `cut` defines where the distribution is truncated.
        """

        data_left = data[data <= cut]
        data_right = data[data > cut]
        N_all = len(data)
        N_left = len(data_left)

        # three parameters will be fitted: dof, location, scale
        p_start = [2., -1., 1.]
        p_bounds = [(0., None),  # dof > 0
                    (None, -0),  # location < 0 for 'truncated' effect
                    (1e-5, 1e5)]  # shape ~ free

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
        """ Probability density function.
        """
        m_flat = (0. <= x) & (x <= self._cut)
        m_chi2 = (x > self._cut)

        x = np.asarray(x)
        r = np.zeros_like(x)
        if self._cut == 0.:
            r[m_flat] = self._f.cdf(self._cut)
        else:
            r[m_flat] = self._f.cdf(self._cut) / self._cut
        r[m_chi2] = self._f.pdf(x[m_chi2])
        return r

    def cdf(self, x):
        """ Cumulative distribution function.
        """
        return self._f.cdf(x)

    def sf(self, x):
        """ Survival function.
        """
        return self._f.sf(x)

    def isf(self, x):
        """ Inverse survival function.
        """
        return self._f.isf(x)

    def ppf(self, x):
        """ Percent point function.
        """
        return self._f.ppf(x)

    def __str__(self):
        return 'Left-truncated Chi^2 Distribution:\n' + \
               '\t DoF      = {0:7.2f}\n'.format(self._f.args[0]) + \
               '\t Location = {0:7.2f}\n'.format(self._f.args[1]) + \
               '\t Scale    = {0:7.2f}\n'.format(self._f.args[2]) + \
               '\t KS       = {0:7.2%}'.format(self._ks)

class Double_Chi2(object):
    """ A class similar to the ones from scipy.stats
       allowing to fit left-truncated chi^2 distributions.
    """

    def __init__(self, data):
        # p_start = [4., -1., 1.]

        med = np.median(data)

        data_right = data[data > med]

        data_left = data[data <= med]

        p_start = [2.]
        p_bounds = [(0., None),  # dof > 0
                    # (None, -0),  # location < 0 for 'truncated' effect
                    #  shape ~ free
                    ]
                    # (1e-5, 1e5)]  # shape ~ free

        def func(p):
            loglh = scipy.stats.chi2(p[0], loc=med, scale=1).logpdf(
                data_right).sum()
            # print loglh, p
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        # print res

        self._cut = 0.
        self._res = res
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self._df = res.x[0]
        self._loc = med
        self._scale = 1.

        def func(p):
            loglh = scipy.stats.chi2(p[0], loc=-med, scale=1).logpdf(
                -data_left).sum()
            # print loglh, p
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        self.res_left = res
        self._f_left = scipy.stats.chi2(*res.x)

class Chi2_one_side:

    def __init__(self, data):
        p_start = [2., -1., 1.]
        p_start = [1.3]
        p_bounds = [(0., None),  # dof > 0
                    ]
                    # (1e-5, 1e5)]  # shape ~ free

        def func(p):
            loglh = scipy.stats.chi2(p[0], loc=0., scale=1.).logpdf(
                data).sum()
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        self._cut = 0.
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self._res = res


class Chi2_one_side_free:

    def __init__(self, data):
        # p_start = [4., -1., 1.]
        p_start = [2., -1., 1.]
        p_bounds = [(0., None),  # dof > 0
                    (None, -0),  # location < 0 for 'truncated' effect
                    (1e-5, 1e5)# shape ~ free
                    ]
                    # (1e-5, 1e5)]  # shape ~ free

        def func(p):
            loglh = scipy.stats.chi2(p[0], loc=p[1]-1e-8, scale=p[2]).logpdf(
                data).sum()
            # print loglh, p
            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        # print res

        self._cut = 0.
        self._res = res
        self._f = scipy.stats.chi2(*res.x)
        self._ks = scipy.stats.kstest(data, self._f.cdf)[0]
        self.ndof = res.x[0]
        self.loc = res.x[1]
        self.scale = res.x[2]


def fit_background_ts(ts_array, ts_type):

    mask = ts_array > 0.0
    frac_over = float(len(ts_array[mask])) / (float(len(ts_array)))

    if ts_type == "Flare":

        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        chi2 = Chi2_LeftTruncated(ts_array)

        if chi2._res.success:

            frac_over = 1.

            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

        else:
            chi2 = Chi2_one_side_free(ts_array[ts_array > 0.])
            df = chi2._f.args[0]
            loc = chi2._f.args[1]
            scale = chi2._f.args[2]

            if not chi2._res.success:
                chi2 = Chi2_one_side(ts_array[ts_array > 0.])
                df = chi2._f.args[0]
                loc = 0.
                scale = 1.

    elif ts_type == "Fit Weights":

        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        chi2 = Chi2_one_side_free(ts_array[ts_array > 0.])

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
                chi2 = Chi2_one_side(ts_array[ts_array > 0.])
                df = chi2._f.args[0]
                loc = 0.
                scale = 1.

            else:
                frac_over = 1.

    elif ts_type in ["Standard", "Negative n_s"]:

        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        chi2 = Chi2_one_side(ts_array[ts_array > 0.])

        df = chi2._f.args[0]
        loc = 0.
        scale = 1.

    else:
        raise Exception("ts_type " + str(ts_type) + " not recognised!")

    return df, loc, scale, frac_over


def plot_expanded_negative(ts_array, path):

    plt.figure()
    plt.hist([ts_array[ts_array > 0], ts_array[ts_array < 0]],
             bins=n_bins, lw=2, histtype='step',
             color=['black', "grey"],
             label=['TS > 0', "TS <= 0"],
             normed=True,
             stacked=True)

    med = np.median(ts_array)

    chi2 = Double_Chi2(ts_array)

    # frac_over = 0.5

    # x_range = np.linspace(med, max(ts_array), 100)
    #
    # plt.plot(x_range, frac_over * chi2._f.pdf(x_range),
    #          color="blue", label=r"$\chi^{2}$ Distribution")

    # x_range = np.linspace(min(ts_array), med, 100)
    #
    # plt.plot(x_range, frac_over * chi2._f_left.pdf(-x_range),
    #          color="blue")

    plt.yscale("log")
    plt.xlabel(r"Test Statistic ($\lambda$)")
    plt.legend(loc="upper right")
    plt.savefig(path[:-4] + "_expanded.pdf")
    plt.close()

def plot_background_ts_distribution(ts_array, path, ts_type="Standard",
                                    ts_val=None):

    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

    ts_array = np.array(ts_array)
    ts_array = ts_array[~np.isnan(ts_array)]

    if np.sum(np.isnan(ts_array)) > 0:
        print "TS distribution has", np.sum(np.isnan(ts_array)), "nan entries."

    if np.median(ts_array) < 0.:
        plot_expanded_negative(ts_array, path)

    fig = plt.figure()

    df, loc, scale, frac_over = fit_background_ts(ts_array, ts_type)

    frac_under = 1 - frac_over

    five_sigma = (raw_five_sigma - frac_under) / (1. - frac_under)

    plt.axhline(frac_over * (1 - five_sigma), color="r", linestyle="--")

    max_ts = np.max(ts_array)

    disc_potential = scipy.stats.chi2.ppf(five_sigma, df, loc, scale)

    x_range = np.linspace(0., max(max_ts, disc_potential), 100)

    plt.plot(x_range, frac_over * scipy.stats.chi2.pdf(x_range, df, loc, scale),
             color="blue", label=r"$\chi^{2}$ Distribution")

    def integral(x):

        return (frac_under * np.sign(x) + frac_over *
                (scipy.stats.chi2.cdf(x, df, loc, scale)))

    plt.plot(x_range, 1. - integral(x_range), color="green", linestyle="--",
             label=r"1 - $\int f(x)$ (p-value)")

    plt.axvline(disc_potential, color="r", label=r"5 $\sigma$ Threshold")

    if ts_val is not None:
        print "\n"
        print "Quantifying TS:", "{:.2f}".format(ts_val)

        if ts_val > np.median(ts_array):

            val = (ts_val - frac_under) / (1. - frac_under)

            cdf = frac_under + frac_over*scipy.stats.chi2.cdf(val, df, loc, scale)

            sig = norm.ppf(cdf)

        else:
            cdf = 0.
            sig = 0.

        print "Pre-trial P-value is", "{:.2E}".format(1-cdf), 1-cdf
        print "Significance is", "{:.2f}".format(sig), "Sigma"
        print "\n"

        plt.axvline(ts_val, color="purple",
                    label="{:.2f}".format(ts_val) + " TS/" +
                    "{:.2f}".format(sig) + r" $\sigma$")

    else:
        plt.annotate(
            '{:.1f}'.format(100 * frac_under) + "% of data in delta. \n" +
            r"$\chi^{2}$ Distribution:" + "\n   * d.o.f.=" + \
            '{:.2f}'.format(df) + ",\n  * loc=" + '{:.2f}'.format(loc) + \
            " \n * scale=" + '{:.2f}'.format(scale),
            xy=(0.1, 0.2), xycoords="axes fraction", fontsize=8)

    yrange = min(1. / (float(len(ts_array)) * n_bins),
                 scipy.stats.chi2.pdf(disc_potential, df, loc, scale))

    plt.yscale("log")
    plt.xlabel(r"Test Statistic ($\lambda$)")
    plt.legend(loc="upper right")
    plt.savefig(path)
    plt.close()

    return disc_potential


def plot_fit_results(results, path, labels, inj=None):

    results = np.array(results)

    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

    n_dim = len(results)

    try:

        fig = plt.figure()

        fig.set_size_inches(7, n_dim*3)
        fig.subplots_adjust(hspace=.5)

        for i, row in enumerate(results):

            weights = np.ones(len(row))/float(len(row))

            label = labels[i]

            plt.subplot(n_dim, 1, i+1)
            plt.hist(row, histtype="step", weights=weights,
                     bins=100, color="blue")
            plt.axvline(np.median(row), linestyle="--", color="blue",
                        label="Median")
            plt.title(label)

            if inj is not None and label == "n_s":
                n_s = 0
                for val in inj.itervalues():
                    n_s += val["n_s"]
                plt.axvline(n_s, linestyle="--", color="orange", label="Injection")

            elif inj is not None and "(" in label:

                keys = [x[:-1] for x in label.split("(")]
                val = inj[keys[1]][keys[0]]
                plt.axvline(val, linestyle="--", color="orange",
                            label="Injection")

            elif inj is not None:
                val = inj.itervalues().next()[label]
                plt.axvline(val, linestyle="--", color="orange",
                            label="Injection")

            plt.legend()

        plt.savefig(path)
        plt.close()

    except ValueError:
        pass

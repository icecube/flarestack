import numpy as np
import os
from shared import plots_dir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize, scipy.stats


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
        """ Fit the given ensemble of measurements with a chi^2 function.

        `data` is a list of test statistics values.
        `cut` defines where the distribution is truncated.
        """

        print "Running Double Chi2"

        data_left = data[data <= 0.]
        data_right = data[data > 0.]
        N_all = len(data)

        print N_all, len(data_left), len(data_right)

        f_under = float(len(data_left))/float(N_all)
        f_over = float(len(data_right)) / float(N_all)

        print f_under, f_over, f_under + f_over

        N_left = len(data_left)

        # four parameters will be fitted: dof x 2, scale x 2
        p_start = [2., 2., f_under, f_over]
        p_bounds = [(1e-9, None),
                    (1e-9, None),# dof > 0
                    (1e-5, 1e5),
                    (1e-5, 1e5)]  # shape ~ free

        # define the fit function: likelihood for 2 chi^2 distributions,
        # one for positive values and another for negative

        def func(p):
            #
            # print p

            # left = scipy.stats.chi2(p[0], loc=0., scale=p[2])
            right = scipy.stats.chi2(p[1], loc=0, scale=p[3])

            # loglh = np.log(
            #     f_under * left.pdf(-data_left).sum() +
            #     f_over * right.pdf(data_right).sum()
            # )

            loglh = right.logpdf(data_right).sum()
            # print loglh

            return -loglh

        res = scipy.optimize.minimize(func, x0=p_start, bounds=p_bounds)

        # if not res.success:
        #     print 'Chi2 fit did not converge! Result is likely garbage.'

        self._f_left = scipy.stats.chi2(res.x[0], loc=0., scale=res.x[2])
        self._f_right = scipy.stats.chi2(res.x[1], loc=0., scale=res.x[3])
        self._res = res
        self._frac_over = f_over
        self.frac_under = f_under

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

def plot_background_ts_distribution(ts_array, path, ts_type="Standard"):
    ts_array = np.array(ts_array)
    ts_array = ts_array[~np.isnan(ts_array)]
    med = np.median(ts_array)

    raw_five_sigma = 0.999999713349

    mask = ts_array > 0.

    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

    frac_over = float(len(ts_array[mask])) / (float(len(ts_array)))

    if ts_type == "Flare":

        n_bins = 100

        plt.figure()
        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        chi2 = Chi2_LeftTruncated(ts_array)
        frac_over = float(len(ts_array[mask])) / (float(len(ts_array)))

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

        frac_under = 1. - frac_over
        five_sigma = (raw_five_sigma - frac_under) / (1. - frac_under)

    elif ts_type == "Fit Weights":

        n_bins = 100

        plt.figure()
        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        # chi2 = Chi2_LeftTruncated(ts_array)
        chi2 = Chi2_one_side_free(ts_array[ts_array > 0.])
        # frac_over = float(len(ts_array[mask])) / (float(len(ts_array)))

        # print chi2._res.success
        # raw_input("prompt")

        if chi2._res.success:

            # frac_over = 1.

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

        frac_under = 1. - frac_over
        five_sigma = (raw_five_sigma - frac_under) / (1. - frac_under)

    elif ts_type == "Standard":
        frac_under = 1. - frac_over

        n_bins = 100

        plt.figure()
        plt.hist([ts_array[mask], np.zeros(np.sum(~mask))],
                 bins=n_bins, lw=2, histtype='step',
                 color=['black', "grey"],
                 label=['TS > 0', "TS <= 0"],
                 normed=True,
                 stacked=True)

        five_sigma = (raw_five_sigma - frac_under) / (1. - frac_under)

        chi2 = Chi2_one_side(ts_array[ts_array > 0.])

        df = chi2._f.args[0]
        loc = 0.
        scale = 1.

    else:
        raise Exception("ts_type " + str(ts_type) + " not recognised!")
        # except IndexError:
        #     scale = 1.
    #
    # print ts_type
    # print chi2._res
    # print Chi2_one_side(ts_array[ts_array > 0])._res
    # print Chi2_one_side(ts_array[~(ts_array < 0)])._res
    # print Chi2_one_side_free(ts_array[~(ts_array < 0)])._res
    # print Chi2_LeftTruncated(ts_array)._res
    # raw_input("prompt")
    # df = 1.31
    # loc = 0.
    # scale = 1.0

    plt.axhline(frac_over * (1 - five_sigma), color="r", linestyle="--")

    max_ts = np.max(ts_array)

    disc_potential = scipy.stats.chi2.ppf(five_sigma, df, loc, scale)

    xrange = np.linspace(0., max(max_ts, disc_potential), 100)

    plt.plot(xrange, frac_over * scipy.stats.chi2.pdf(xrange, df, loc, scale),
             color="blue", label=r"$\chi^{2}$ Distribution")

    def integral(x):

        return (frac_under * np.sign(x) + frac_over *
                (scipy.stats.chi2.cdf(x, df, loc, scale)))

    plt.plot(xrange, 1. - integral(xrange), color="green", linestyle="--",
             label=r"1 - $\int f(x)$")

    plt.axvline(disc_potential, color="r", label=r"5 $\sigma$ Threshold")
    plt.annotate(
        '{:.1f}'.format(100 * frac_under) + "% of data in delta. \n" +
        r"$\chi^{2}$ Distribution:" + "\n   * d.o.f.=" + \
        '{:.2f}'.format(df) + ",\n  * loc=" + '{:.2f}'.format(loc) + " \n"
        " * scale=" + '{:.2f}'.format(scale),
        xy=(0.3, 0.8), xycoords="axes fraction")

    yrange = min(1. / (float(len(ts_array)) * n_bins),
                 scipy.stats.chi2.pdf(disc_potential, df, loc, scale))

    # plt.ylim((0.5 * yrange, max()))
    plt.yscale("log")
    # plt.grid()
    plt.xlabel(r"Test Statistic ($\lambda$)")
    plt.legend()
    plt.savefig(path)

    # if

    plt.close()
    #
    # print chi2, disc_potential, 1-frac_over
    # raw_input("prompt")

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

        # print len(results), labels
        # print [x[:5] for x in results]
        # fs = results[2]
        # fe = results[3]
        # fl = results[4]
        #
        # mask = (fe > 55690.) & (fe < 55700)
        #
        # print fs[mask][:5], fe[mask][:5], fl[mask][:5]
        #
        # raw_input("prompt")

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

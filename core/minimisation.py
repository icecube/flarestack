import numpy as np
import resource
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import scipy.optimize
from core.injector import Injector
from core.llh import LLH, FlareLLH
from core.ts_distributions import plot_background_ts_distribution


class MinimisationHandler:
    """Generic Class to handle both dataset creation and llh minimisation from
    experimental data and Monte Carlo simulation. Initilialised with a set of
    IceCube datasets, a list of sources, and independent sets of arguments for
    the injector and the likelihood.
    """
    n_trials = 1000

    def __init__(self, name, datasets, sources, inj_kwargs, llh_kwargs,
                 scale=1.):

        self.name = name
        self.injectors = dict()
        self.llhs = dict()
        self.seasons = datasets
        self.sources = sources

        self.inj_kwargs = inj_kwargs
        self.llh_kwargs = llh_kwargs

        # Checks if the code should search for flares. By default, this is
        # not done.
        try:
            self.flare = llh_kwargs["Flare Search?"]
        except KeyError:
            self.flare = False

        if self.flare:
            self.run = self.run_flare
        else:
            self.run = self.run_stacked

        # For each season, we create an independent injector and a
        # likelihood, using the source list along with the sets of energy/time
        # PDFs provided in inj_kwargs and llh_kwargs.
        for season in self.seasons:
            self.injectors[season["Name"]] = Injector(season, sources,
                                                      **inj_kwargs)

            if not self.flare:
                self.llhs[season["Name"]] = LLH(season, sources, **llh_kwargs)
            else:
                self.llhs[season["Name"]] = FlareLLH(season, sources,
                                                     **llh_kwargs)

        # The default value for n_s is 1. It can be between 0 and 1000.
        p0 = [1.]
        bounds = [(0, 1000.)]
        names = ["n_s"]

        # If weights are to be fitted, then each source has an independent
        # n_s in the same 0-1000 range.
        if "Fit Weights?" in llh_kwargs.keys():
            if llh_kwargs["Fit Weights?"]:
                p0 = [1. for x in sources]
                bounds = [(0, 1000.) for x in sources]
                names = ["n_s (" + x["Name"] + ")" for x in sources]

        # If gamma is to be included as a fit parameter, then its default
        # value if 2, and it can range between 1 and 4.
        if "Fit Gamma?" in llh_kwargs.keys():
            if llh_kwargs["Fit Gamma?"]:
                p0.append(2.)
                bounds.append((1., 4.))
                names.append("Gamma")

        self.p0 = p0
        self.bounds = bounds
        self.param_names = names

        # Sets the default flux scale for finding sensitivity
        # Default value is 1 (Gev)^-1 (cm)^-2 (s)^-1
        self.scale = scale
        self.bkg_ts = None

    def run_stacked(self, n=n_trials, scale=1):

        param_vals = [[] for x in self.p0]
        ts_vals = []
        flags = []

        print "Generating", n, "trials!"

        for i in tqdm(range(int(n))):
            f = self.run_trial(scale)

            res = scipy.optimize.fmin_l_bfgs_b(
                f, self.p0, bounds=self.bounds, approx_grad=True)

            flag = res[2]["warnflag"]
            # If the minimiser does not converge, repeat with brute force
            if flag > 0:
                try:
                    res = scipy.optimize.brute(f, ranges=self.bounds,
                                               full_output=True)

                except KeyError:
                    res = None

            if res is not None:
                vals = res[0]
                ts = -res[1]

                for j, val in enumerate(vals):
                    param_vals[j].append(val)

                ts_vals.append(float(ts))
                flags.append(flag)

        mem_use = str(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
        print ""
        print 'Memory usage max: %s (Gb)' % mem_use

        n_inj = 0
        for inj in self.injectors.itervalues():
            for val in inj.ref_fluxes.itervalues():
                n_inj += val
        print ""
        print "Injected with an expectation of", n_inj, "events."

        print ""
        print "FIT RESULTS:"
        print ""

        for i, param in enumerate(param_vals):
            print "Parameter", self.param_names[i], ":", np.mean(param), \
                np.median(param), np.std(param)
        print "Test Statistic:", np.mean(ts_vals), np.median(ts_vals), np.std(
            ts_vals)
        print ""

        print "FLAG STATISTICS:"
        for i in sorted(np.unique(flags)):
            print "Flag", i, ":", flags.count(i)

        bkg_median = 0.

        frac_over = np.sum(np.array(ts_vals) > bkg_median) / float(len(ts_vals))

        print "Fraction of overfluctuations", frac_over

        return ts_vals, param_vals, flags

    def run_trial(self, scale=1):

        llh_functions = dict()

        for season in self.seasons:
            dataset = self.injectors[season["Name"]].create_dataset(scale)
            llh_f = self.llhs[season["Name"]].create_llh_function(dataset)
            llh_functions[season["Name"]] = llh_f

        def f_final(params):

            # Creates a matrix fixing the fraction of the total signal that
            # is expected in each Source+Season pair. The matrix is
            # normalised to 1, so that for a given total n_s, the expectation
            # for the ith season for the jth source is given by:
            #  n_exp = n_s * weight_matrix[i][j]

            weights_matrix = np.ones([len(self.seasons), len(self.sources)])

            for i, season in enumerate(self.seasons):
                llh = self.llhs[season["Name"]]
                acc = llh.acceptance(self.sources, params)

                time_weights = []

                for source in self.sources:

                    time_weights.append(llh.time_pdf.effective_injection_time(
                        source))

                w = acc * self.sources["weight_distance"] * np.array(
                    time_weights)

                w = w[:, np.newaxis]

                for j, ind_w in enumerate(w):
                    weights_matrix[i][j] = ind_w

            weights_matrix /= np.sum(weights_matrix)

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, (name, f) in enumerate(llh_functions.iteritems()):
                w = weights_matrix[i][:, np.newaxis]
                ts_val += f(params, w)

            return -ts_val

        return f_final

    def run_flare(self, n=n_trials, scale=1):

        print "Running", n, "trials"

        results = {
            "Stacked": []
        }

        for source in self.sources:
            results[source["Name"]] = []

        for i in tqdm(range(int(n))):

            datasets = dict()

            full_data = dict()

            for season in self.seasons:
                data = self.injectors[season["Name"]].create_dataset(scale)
                llh = self.llhs[season["Name"]]

                full_data[season["Name"]] = data

                for source in self.sources:
                    mask = llh.select_spatially_coincident_data(data, [source])
                    spatial_coincident_data = data[mask]

                    t_mask = np.logical_and(
                        np.greater(spatial_coincident_data["timeMJD"],
                                   source["Start Time (MJD)"]),
                        np.less(spatial_coincident_data["timeMJD"],
                                source["End Time (MJD)"])
                    )

                    coincident_data = spatial_coincident_data[t_mask]

                    name = source["Name"]
                    if name not in datasets.keys():
                        datasets[name] = dict()

                    if len(coincident_data) > 0:

                        new_entry = dict(season)
                        new_entry["Coincident Data"] = spatial_coincident_data
                        significant = llh.find_significant_events(
                            coincident_data, source)
                        new_entry["Significant Times"] = significant["timeMJD"]
                        new_entry["N_all"] = len(data)

                        datasets[name][season["Name"]] = new_entry

            stacked_ts = 0.0

            for (source, source_dict) in datasets.iteritems():

                src = self.sources[self.sources["Name"] == source]

                all_times = []
                n_tot = 0
                for season_dict in source_dict.itervalues():
                    new_times = season_dict["Significant Times"]
                    all_times.extend(new_times)
                    n_tot += len(season_dict["Coincident Data"])

                all_times = np.array(sorted(all_times))

                # print "In total", len(all_times), "of", n_tot

                # Minimum flare duration (days)
                min_flare = 1.

                pairs = []

                for x in all_times:
                    for y in all_times:
                        if y > (x + min_flare):
                            pairs.append((x, y))
                #
                # print pairs
                # raw_input("prompt")

                # all_pairs = [(x, y) for x in all_times for y in all_times if y > x]

                # print "This gives", len(pairs), "possible pairs out of",
                # print len(all_pairs), "pairs."

                if len(pairs) > 0:

                    all_res = []
                    all_ts = []

                    for pair in pairs:
                        t_start = pair[0]
                        t_end = pair[1]

                        w = np.ones(len(source_dict))

                        llhs = dict()

                        for i, season_dict in enumerate(source_dict.itervalues()):
                            coincident_data = season_dict["Coincident Data"]
                            data = full_data[season_dict["Name"]]
                            flare_veto = np.logical_or(
                                np.less(coincident_data["timeMJD"], t_start),
                                np.greater(coincident_data["timeMJD"], t_end))

                            if np.sum(~flare_veto) > 0:

                                t_s = max(t_start, season_dict["Start (MJD)"])
                                t_e = min(t_end, season_dict["End (MJD)"])
                                flare_length = t_e - t_s

                                t_s_min = max(src["Start Time (MJD)"][0],
                                              season_dict["Start (MJD)"])
                                t_e_max = min(src["End Time (MJD)"][0],
                                              season_dict["End (MJD)"])
                                max_flare = t_e_max - t_s_min

                                full_flare_veto = np.logical_or(
                                    np.less(data["timeMJD"], t_s_min),
                                    np.greater(data["timeMJD"], t_e_max))

                                n_all = len(data[~full_flare_veto])

                                marginalisation = flare_length / max_flare

                                llh_kwargs = dict(self.llh_kwargs)
                                llh_kwargs["LLH Time PDF"]["Name"] = "FixedBox"
                                llh_kwargs["LLH Time PDF"]["Start Time (" \
                                                           "MJD)"] = t_s
                                llh_kwargs["LLH Time PDF"]["End Time (" \
                                                           "MJD)"] = t_e
                                llh_kwargs["LLH Time PDF"]["Bkg Start Time (" \
                                                           "MJD)"] = t_s_min
                                llh_kwargs["LLH Time PDF"]["Bkg End Time (" \
                                                           "MJD)"] = t_e_max

                                llh = self.llhs[season["Name"]]

                                flare_llh = llh.create_flare(season_dict, src,
                                                             **llh_kwargs)

                                flare_f = flare_llh.create_llh_function(
                                    coincident_data, flare_veto, n_all,
                                    marginalisation)

                                llhs[season_dict["Name"]] = {
                                    "llh": flare_llh,
                                    "f": flare_f,
                                    "flare length": flare_length
                                }

                        # From here, we have normal minimisation behaviour

                        def f_final(params):

                            weights_matrix = np.ones(len(llhs))

                            for i, llh_dict in enumerate(llhs.itervalues()):
                                T = llh_dict["flare length"]
                                acc = llh.acceptance(src, params)
                                weights_matrix[i] = T * acc

                            weights_matrix /= np.sum(weights_matrix)

                            ts = 0

                            for i, llh_dict in enumerate(llhs.itervalues()):
                                w = weights_matrix[i]
                                ts += llh_dict["f"](params, w)

                            return -ts

                        res = scipy.optimize.fmin_l_bfgs_b(
                            f_final, self.p0, bounds=self.bounds,
                            approx_grad=True)

                        all_res.append(res)
                        all_ts.append(-res[1])

                    max_ts = max(all_ts)
                    stacked_ts += max_ts
                    index = all_ts.index(max_ts)

                    results[source].append([max_ts, pairs[index],
                                                 all_res[index]])

            results["Stacked"].append(stacked_ts)

        full_ts = results["Stacked"]

        print "Combined Test Statistic:"
        print np.mean(full_ts), np.median(full_ts), np.std(
              full_ts)
        # plot_background_ts_distribution(full_ts)

        print ""

        for source in self.sources:
            print "Results for", source["Name"]

            combined_res = results[source["Name"]]

            full_ts = np.array([x[0] for x in combined_res])

            full_params = np.array([x[2][0] for x in combined_res ])

            starts = np.array([x[1][0] for x in combined_res])

            ends = np.array([x[1][1] for x in combined_res ])

            lengths = np.array([x[1][1] - x[1][0] for x in combined_res])

            for i, column in enumerate(full_params.T):
                print "Param", i, np.mean(column), np.median(column), np.std(column)
            print "Window length:", np.mean(lengths), np.median(lengths), np.std(
                lengths)

            print "Window start:", np.mean(starts), np.median(starts), np.std(
                starts)
            print "Window end:", np.mean(ends), np.median(ends), np.std(ends)

            print "Source:", source["Start Time (MJD)"],
            print source["End Time (MJD)"], source["Ref Time (MJD)"]

            print "Test Statistic", np.mean(full_ts), np.median(full_ts), np.std(
                  full_ts), "\n"

            # plot_background_ts_distribution(full_ts)

            # bkg_median = 5.51
            #
            # frac_over = np.sum(full_ts > bkg_median) / float(n)
            #
            # print "Fraction of overfluctuations", frac_over

    def scan_likelihood(self, scale=1):

        f = self.run(scale)

        n_range = np.linspace(1, 200, 1e4)
        y = []

        for n in tqdm(n_range):
            new = f([n])
            try:
                y.append(new[0][0])
            except IndexError:
                y.append(new)

        plt.figure()
        plt.plot(n_range, y)
        plt.savefig("llh_scan.pdf")
        plt.close()

        min_y = np.min(y)
        print "Minimum value of", min_y,

        min_index = y.index(min_y)
        min_n = n_range[min_index]
        print "at", min_n

        l_y = np.array(y[:min_index])
        try:
            l_y = min(l_y[l_y > (min_y + 0.5)])
            l_lim = n_range[y.index(l_y)]
        except ValueError:
            l_lim = 0

        u_y = np.array(y[min_index:])
        try:
            u_y = min(u_y[u_y > (min_y + 0.5)])
            u_lim = n_range[y.index(u_y)]
        except ValueError:
            u_lim = ">" + str(max(n_range))

        print "One Sigma interval between", l_lim, "and", u_lim

    def bkg_trials(self):
        """Generate 1000 Background Trials, and plots the distribution. Also
        fits a Chi-squared distribution to the data, accounting for the fact
        that the Test Statistic distribution is truncated at 0.

        :return: Array of Test Statistic values
        """
        print "Generating background trials"

        bkg_ts, params, flags = self.run(100, 0.0)

        ts_array = np.array(bkg_ts)

        frac = float(len(ts_array[ts_array <= 0])) / (float(len(ts_array)))

        print "Fraction of underfluctuations is", frac

        plot_background_ts_distribution(ts_array)

        self.bkg_ts = ts_array

        return ts_array

    def find_sensitivity(self):

        if self.bkg_ts is None:
            self.bkg_trials()

        bkg_median = np.median(self.bkg_ts)

        ts = self.run_trials(100, scale=self.scale)[0]
        frac_over = np.sum(ts > bkg_median) / float(len(ts))

        if frac_over < 0.95:
            rescale = 10
        else:
            rescale = 0.1

        converge = False

        # while not conver

import numpy as np
import resource
import random
import os
import argparse
import cPickle as Pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from tqdm import tqdm
import scipy.optimize
from core.injector import Injector
from core.llh import LLH, FlareLLH
from core.ts_distributions import plot_background_ts_distribution, \
    plot_fit_results
from shared import name_pickle_output_dir, fit_setup, inj_dir_name,\
    plot_output_dir


class MinimisationHandler:
    """Generic Class to handle both dataset creation and llh minimisation from
    experimental data and Monte Carlo simulation. Initilialised with a set of
    IceCube datasets, a list of sources, and independent sets of arguments for
    the injector and the likelihood.
    """
    n_trials_default = 1000

    def __init__(self, mh_dict):

        sources = np.sort(np.load(mh_dict["catalogue"]), order="Distance (Mpc)")

        self.name = mh_dict["name"]
        self.pickle_output_dir = name_pickle_output_dir(self.name)
        self.injectors = dict()
        self.llhs = dict()
        self.seasons = mh_dict["datasets"]
        self.sources = sources

        self.inj_kwargs = mh_dict["inj kwargs"]
        self.llh_kwargs = mh_dict["llh kwargs"]

        # Checks if the code should search for flares. By default, this is
        # not done.
        try:
            self.flare = self.llh_kwargs["Flare Search?"]
        except KeyError:
            self.flare = False

        if self.flare:
            self.run = self.run_flare
        else:
            self.run = self.run_stacked

        try:
            self.brute = self.llh_kwargs["Brute Seed?"]
        except KeyError:
            self.brute = False

        try:
            self.fit_weights = self.llh_kwargs["Fit Weights?"]
        except KeyError:
            self.fit_weights = False

        print "Fit Weights?", self.fit_weights

        if self.fit_weights:
            self.run_trial = self.run_fit_weight_trial
        else:
            self.run_trial = self.run_fixed_weight_trial

        # For each season, we create an independent injector and a
        # likelihood, using the source list along with the sets of energy/time
        # PDFs provided in inj_kwargs and llh_kwargs.
        for season in self.seasons:

            if not self.flare:
                self.llhs[season["Name"]] = LLH(season, sources,
                                                **self.llh_kwargs)
            else:
                self.llhs[season["Name"]] = FlareLLH(season, sources,
                                                     **self.llh_kwargs)

            self.injectors[season["Name"]] = Injector(season, sources,
                                                  **self.inj_kwargs)

        p0, bounds, names = fit_setup(self.llh_kwargs, sources, self.flare)

        self.p0 = p0
        self.bounds = bounds
        self.param_names = names

        # Sets the default flux scale for finding sensitivity
        # Default value is 1 (Gev)^-1 (cm)^-2 (s)^-1
        self.bkg_ts = None

        # self.clean_true_param_values()

    def clear(self):

        self.injectors.clear()
        self.llhs.clear()

        del self

    def dump_results(self, results, scale, seed):
        """Takes the results of a set of trials, and saves the dictionary as
        a pickle pkl_file. The flux scale is used as a parent directory, and the
        pickle pkl_file itself is saved with a name equal to its random seed.

        :param results: Dictionary of Minimisation results from trials
        :param scale: Scale of inputted flux
        :param seed: Random seed used for running of trials
        """

        write_dir = self.pickle_output_dir + '{0:.4G}'.format(scale) + "/"

        # Tries to create the parent diretory, unless it already exists
        try:
            os.makedirs(write_dir)
        except OSError:
            pass

        file_name = write_dir + str(seed) + ".pkl"

        print "Saving to", file_name

        with open(file_name, "wb") as f:
            Pickle.dump(results, f)

    def dump_injection_values(self, scale):

        inj_dict = dict()
        for source in self.sources:
            name = source["Name"]
            n_inj = 0
            for inj in self.injectors.itervalues():
                n_inj += inj.ref_fluxes[scale][name]

            default = {
                "n_s": n_inj
            }

            if "Gamma" in self.param_names:
                default["Gamma"] = self.inj_kwargs["Injection Energy PDF"][
                    "Gamma"]

            if self.flare:
                fs = [inj.time_pdf.sig_t0(source)
                      for inj in self.injectors.itervalues()]
                true_fs = min(fs)
                fe = [inj.time_pdf.sig_t1(source)
                      for inj in self.injectors.itervalues()]
                true_fe = max(fe)

                true_l = true_fe - true_fs

                sim = [
                    list(np.random.uniform(true_fs, true_fe,
                                      np.random.poisson(n_inj)))
                    for i in range(1000)
                ]

                s = []
                e = []
                l = []

                for data in sim:
                    if data != []:
                        s.append(min(data))
                        e.append(max(data))
                        l.append(max(data) - min(data))

                if len(s) > 0:
                    med_s = np.median(s)
                    med_e = np.median(e)
                    med_l = np.median(l)
                else:
                    med_s = np.nan
                    med_e = np.nan
                    med_l = np.nan

                print med_s, med_e, med_l

                default["Flare Start"] = med_s
                default["Flare End"] = med_e
                default["Flare Length"] = med_l

            inj_dict[name] = default

        inj_dir = inj_dir_name(self.name)

        # Tries to create the parent directory, unless it already exists
        try:
            os.makedirs(inj_dir)
        except OSError:
            pass

        file_name = inj_dir + str(scale) + ".pkl"
        with open(file_name, "wb") as f:
            Pickle.dump(inj_dict, f)

    def iterate_run(self, scale=1, n_steps=5, n_trials=50):

        scale_range = np.linspace(0., scale, n_steps)[1:]

        self.run(n_trials*10, scale=0.0)

        # truth_dict = dict()

        for scale in scale_range:
            self.run(n_trials, scale)
            # truth_dict[scale] = new

    def run_stacked(self, n_trials=n_trials_default, scale=1.):

        seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        param_vals = [[] for x in self.p0]
        ts_vals = []
        flags = []

        print "Generating", n_trials, "trials!"

        for i in range(int(n_trials)):
            raw_f = self.run_trial(scale)

            def llh_f(scale):
                return -np.sum(raw_f(scale))

            if self.brute:

                brute_range = [
                    (max(x, -30), min(y, 30)) for (x, y) in self.bounds]

                start_seed = scipy.optimize.brute(
                    llh_f, ranges=brute_range, finish=None, Ns=40)
            else:
                start_seed = self.p0

            res = scipy.optimize.minimize(
                llh_f, start_seed, bounds=self.bounds)

            flag = res.status
            vals = res.x
            # ts = -res.fun
            # If the minimiser does not converge, repeat with brute force
            if flag == 1:

                vals = scipy.optimize.brute(llh_f, ranges=self.bounds,
                                            finish=None)
                # ts = -llh_f(vals)

            if res is not None:

                for j, val in enumerate(list(vals)):
                    param_vals[j].append(val)

                best_llh = raw_f(vals)

                if self.fit_weights:

                    ts = np.sum([
                        llh_val * np.sign(vals[j])
                        for j, llh_val in enumerate(best_llh)])

                else:
                    ts = np.sign(vals[0]) * np.sum(best_llh)
                    # print best_llh, vals[0], ts
                    # raw_input("prompt")

                ts_vals.append(ts)
                flags.append(flag)

        mem_use = str(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
        print ""
        print 'Memory usage max: %s (Gb)' % mem_use

        n_inj = 0
        for inj in self.injectors.itervalues():
            for val in inj.ref_fluxes[scale].itervalues():
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

        results = {
            "TS": ts_vals,
            "Parameters": param_vals,
            "Flags": flags
        }

        self.dump_results(results, scale, seed)

        self.dump_injection_values(scale)

    def make_weight_matrix(self, params):

        # Creates a matrix fixing the fraction of the total signal that
        # is expected in each Source+Season pair. The matrix is
        # normalised to 1, so that for a given total n_s, the expectation
        # for the ith season for the jth source is given by:
        #  n_exp = n_s * weight_matrix[i][j]

        src = np.sort(self.sources, order="Distance (Mpc)")
        dist_weight = src["Distance (Mpc)"] ** -2

        weights_matrix = np.ones([len(self.seasons), len(self.sources)])

        for i, season in enumerate(self.seasons):
            llh = self.llhs[season["Name"]]
            acc = []

            time_weights = []

            for source in src:
                time_weights.append(llh.time_pdf.effective_injection_time(
                    source))
                acc.append(llh.acceptance(source, params))

            acc = np.array(acc).T

            w = acc * dist_weight * np.array(time_weights)

            w = w[:, np.newaxis]

            for j, ind_w in enumerate(w.T):
                weights_matrix[i][j] = ind_w

        weights_matrix /= np.sum(weights_matrix)
        return weights_matrix

    def run_fixed_weight_trial(self, scale=1.):

        llh_functions = dict()
        n_all = dict()

        for season in self.seasons:
            dataset = self.injectors[season["Name"]].create_dataset(scale)
            llh_f = self.llhs[season["Name"]].create_llh_function(dataset)
            llh_functions[season["Name"]] = llh_f
            n_all[season["Name"]] = len(dataset)

        def f_final(params):

            weights_matrix = self.make_weight_matrix(params)

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, (name, f) in enumerate(sorted(llh_functions.iteritems())):
                w = weights_matrix[i][:, np.newaxis]
                ts_val += f(params, w)

            return ts_val

        return f_final

    def run_fit_weight_trial(self, scale):
        llh_functions = dict()
        n_all = dict()

        src = np.sort(self.sources, order="Distance (Mpc)")
        dist_weight = src["Distance (Mpc)"] ** -2

        for season in self.seasons:
            dataset = self.injectors[season["Name"]].create_dataset(scale)
            llh_f = self.llhs[season["Name"]].create_llh_function(dataset)
            llh_functions[season["Name"]] = llh_f
            n_all[season["Name"]] = len(dataset)

        def f_final(params):

            # Creates a matrix fixing the fraction of the total signal that
            # is expected in each Source+Season pair. The matrix is
            # normalised to 1, so that for a given total n_s, the expectation
            # for the ith season for the jth source is given by:
            #  n_exp = n_s * weight_matrix[i][j]

            weights_matrix = np.ones([len(self.seasons), len(self.sources)])

            for i, season in enumerate(self.seasons):
                llh = self.llhs[season["Name"]]
                acc = []

                time_weights = []

                for source in src:
                    time_weights.append(
                        llh.time_pdf.effective_injection_time(
                            source))
                    acc.append(llh.acceptance(source, params))

                acc = np.array(acc).T

                w = acc * dist_weight * np.array(time_weights)

                w = w[:, np.newaxis]

                for j, ind_w in enumerate(w.T):
                    weights_matrix[i][j] = ind_w

            for row in weights_matrix.T:
                row /= np.sum(row)

            # weights_matrix /= np.sum(weights_matrix)

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, (name, f) in enumerate(
                    sorted(llh_functions.iteritems())):
                w = weights_matrix[i][:, np.newaxis]
                # print i, name, w, ts_val,
                ts_val += f(params, w)
                # print ts_val

            return ts_val

        return f_final

    def run_flare(self, n_trials=n_trials_default, scale=1.):

        time_key = self.seasons[0]["MJD Time Key"]

        seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        print "Running", n_trials, "trials"

        # Initialises lists for all values that will need ti be stored,
        # in order to verify that the minimisation is working successfuly

        results = {
            "TS": []
        }

        for source in self.sources:
            results[source["Name"]] = {
                "TS": [],
                "Parameters": [[] for x in self.param_names]
            }

        # Loop over trials

        for i in range(int(n_trials)):

            datasets = dict()

            full_data = dict()

            # Loop over each data season

            for season in self.seasons:

                # Generate a scrambled dataset, and save it to the datasets
                # dictionary. Loads the llh for the season.

                data = self.injectors[season["Name"]].create_dataset(scale)
                llh = self.llhs[season["Name"]]

                full_data[season["Name"]] = data

                # Loops over each source in catalogue

                for source in self.sources:

                    # Identify spatially- and temporally-coincident data

                    mask = llh.select_spatially_coincident_data(data, [source])
                    spatial_coincident_data = data[mask]

                    t_mask = np.logical_and(
                        np.greater(
                            spatial_coincident_data[time_key],
                            llh.time_pdf.sig_t0(source)),
                        np.less(
                            spatial_coincident_data[time_key],
                            llh.time_pdf.sig_t1(source))
                    )

                    coincident_data = spatial_coincident_data[t_mask]

                    # Creates empty dictionary to save info

                    name = source["Name"]
                    if name not in datasets.keys():
                        datasets[name] = dict()

                    # If there are events in the window...

                    if len(coincident_data) > 0:

                        new_entry = dict(season)
                        new_entry["Coincident Data"] = coincident_data
                        new_entry["Start (MJD)"] = llh.time_pdf.t0
                        new_entry["End (MJD)"] = llh.time_pdf.t1

                        # Identify significant events (S/B > 1)

                        significant = llh.find_significant_events(
                            coincident_data, source)

                        new_entry["Significant Times"] = significant[time_key]

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

                t_s_min = float(min([llh.time_pdf.sig_t0(source)
                                    for llh in self.llhs.itervalues()]))

                t_e_max = float(max([llh.time_pdf.sig_t1(source)
                                    for llh in self.llhs.itervalues()]))

                max_flare = t_e_max - t_s_min

                pairs = []

                for x in all_times:
                    for y in all_times:
                        if y > (x + min_flare):
                            pairs.append((x, y))

                all_pairs = [(x, y) for x in all_times for y in all_times if y > x]

                if len(pairs) > 0:

                    all_res = []
                    all_ts = []

                    for pair in pairs:
                        t_start = pair[0]
                        t_end = pair[1]

                        flare_length = t_end - t_start

                        overall_marginalisation = flare_length / max_flare
                        w = np.ones(len(source_dict))

                        llhs = dict()

                        for i, (name, season_dict) in enumerate(
                                sorted(source_dict.iteritems())):

                            coincident_data = season_dict["Coincident Data"]

                            data = full_data[name]

                            flare_veto = np.logical_or(
                                np.less(coincident_data[time_key], t_start),
                                np.greater(coincident_data[time_key], t_end))
                            # flare_veto = np.zeros_like(coincident_data["timeMJD"])

                            if np.sum(~flare_veto) > 0:

                                llh = self.llhs[season_dict["Name"]]

                                t_s = min(
                                    max(t_start, season_dict["Start (MJD)"]),
                                    season_dict["End (MJD)"])
                                t_e = max(
                                    min(t_end, season_dict["End (MJD)"]),
                                    season_dict["Start (MJD)"])
                                flare_length = t_e - t_s

                                t_s_min = max(llh.time_pdf.sig_t0(source),
                                              season_dict["Start (MJD)"])
                                t_e_max = min(llh.time_pdf.sig_t1(source),
                                              season_dict["End (MJD)"])
                                max_flare = t_e_max - t_s_min

                                # n_all = len(data[~full_flare_veto])
                                n_all = np.sum(~np.logical_or(
                                    np.less(data[time_key], t_s),
                                    np.greater(data[time_key], t_e)
                                ))

                                if n_all > 0:
                                    pass
                                else:
                                    print t_start, t_end, t_s, t_e, max_flare

                                marginalisation = flare_length / max_flare
                                # marginalisation = 1.

                                llh_kwargs = dict(self.llh_kwargs)
                                llh_kwargs["LLH Time PDF"] = None

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

                            for i, llh_dict in enumerate(
                                    sorted(llhs.itervalues())):
                                T = llh_dict["flare length"]
                                acc = llh.acceptance(src, params)
                                weights_matrix[i] = T * acc

                            weights_matrix /= np.sum(weights_matrix)

                            ts = 2 * np.log(overall_marginalisation)

                            for i, (name, llh_dict) in enumerate(
                                    sorted(llhs.iteritems())):

                                w = weights_matrix[i]

                                if w > 0.:
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

                    best_start = pairs[index][0]
                    best_end = pairs[index][1]

                    best_length = best_end - best_start

                    best = [x for x in all_res[index][0]] + [
                        best_start, best_end, best_length
                    ]

                    for k, val in enumerate(best):
                        results[source]["Parameters"][k].append(val)

                    results[source]["TS"].append(max_ts)

            results["TS"].append(stacked_ts)

        mem_use = str(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
        print ""
        print 'Memory usage max: %s (Gb)' % mem_use

        full_ts = results["TS"]

        print "Combined Test Statistic:"
        print np.mean(full_ts), np.median(full_ts), np.std(
              full_ts)

        for source in self.sources:
            print "Results for", source["Name"]

            combined_res = results[source["Name"]]

            full_ts = combined_res["TS"]

            full_params = np.array(combined_res["Parameters"])

            for i, column in enumerate(full_params):
                print self.param_names[i], ":", np.mean(column),\
                    np.median(column), np.std(column)

            print "Test Statistic", np.mean(full_ts), np.median(full_ts), np.std(
                  full_ts), "\n"

        self.dump_results(results, scale, seed)

        self.dump_injection_values(scale)

    def scan_likelihood(self, scale=1):

        f = self.run_trial(scale)

        def g(x):
            return -np.sum(f(x))
        #
        # g = self.run_trial(scale)
        #
        res = scipy.optimize.minimize(
            g, self.p0, bounds=self.bounds)

        print res.x
        #
        # raw_input("prompt")

        plt.figure(figsize=(8, 4 + 2*len(self.p0)))

        for i, bounds in enumerate(self.bounds):
            plt.subplot(len(self.p0), 1, 1 + i)

            best = list(res.x)

            n_range = np.linspace(max(bounds[0], -100),
                                  min(bounds[1], 100), 1e2)
            y = []

            for n in n_range:

                best[i] = n

                new = g(best)
                try:
                    y.append(new[0][0])
                except IndexError:
                    y.append(new)

            plt.plot(n_range, y)
            plt.xlabel(self.param_names[i])

            print "PARAM:", self.param_names[i]
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

        path = plot_output_dir(self.name) + "llh_scan_" + str(scale) + ".pdf"
        plt.savefig(path)
        plt.close()

        print "Saved to", path

    def check_flare_background_rate(self):

        results = [[] for x in self.seasons]
        total = [[] for x in self.seasons]

        for i in range(int(1000)):

            # Loop over each data season

            for j, season in enumerate(sorted(self.seasons)):

                # Generate a scrambled dataset, and save it to the datasets
                # dictionary. Loads the llh for the season.

                data = self.injectors[season["Name"]].create_dataset(0.0)
                llh = self.llhs[season["Name"]]

                # Loops over each source in catalogue

                for source in np.sorted(self.sources, order="Distance (Mpc)"):

                    # Identify spatially- and temporally-coincident data

                    mask = llh.select_spatially_coincident_data(data, [source])
                    spatial_coincident_data = data[mask]



                    t_mask = np.logical_and(
                        np.greater(spatial_coincident_data["timeMJD"],
                                   llh.time_pdf.sig_t0(source)),
                        np.less(spatial_coincident_data["timeMJD"],
                                llh.time_pdf.sig_t1(source))
                    )

                    coincident_data = spatial_coincident_data[t_mask]
                    total[j].append(len(coincident_data))
                    # If there are events in the window...

                    if len(coincident_data) > 0:

                        # Identify significant events (S/B > 1)

                        significant = llh.find_significant_events(
                            coincident_data, source)

                        results[j].append(len(significant))
                    else:
                        results[j].append(0)

        for j, season in enumerate(sorted(self.seasons)):
            res = results[j]
            tot = total[j]

            print season["Name"],"Significant events", np.mean(res), \
                np.median(res), np.std(res)
            print season["Name"], "All events", np.mean(tot), np.median(tot), \
                np.std(tot)

            llh = self.llhs[season["Name"]]

            for source in self.sources:

                print "Livetime", llh.time_pdf.effective_injection_time(source)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    cfg = parser.parse_args()

    print cfg.file

    with open(cfg.file) as f:
        mh_dict = Pickle.load(f)

    mh = MinimisationHandler(mh_dict)
    mh.iterate_run(mh_dict["scale"], n_steps=mh_dict["n_steps"],
                   n_trials=mh_dict["n_trials"])

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
from shared import name_pickle_output_dir, fit_setup, inj_dir_name


class MinimisationHandler:
    """Generic Class to handle both dataset creation and llh minimisation from
    experimental data and Monte Carlo simulation. Initilialised with a set of
    IceCube datasets, a list of sources, and independent sets of arguments for
    the injector and the likelihood.
    """
    n_trials_default = 1000

    def __init__(self, mh_dict, cleanup=False):

        sources = np.load(mh_dict["catalogue"])

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

        # For each season, we create an independent injector and a
        # likelihood, using the source list along with the sets of energy/time
        # PDFs provided in inj_kwargs and llh_kwargs.
        for season in self.seasons:
            self.injectors[season["Name"]] = Injector(season, sources,
                                                      **self.inj_kwargs)

            if not self.flare:
                self.llhs[season["Name"]] = LLH(season, sources,
                                                **self.llh_kwargs)
            else:
                self.llhs[season["Name"]] = FlareLLH(season, sources,
                                                     **self.llh_kwargs)

        p0, bounds, names = fit_setup(self.llh_kwargs, sources, self.flare)

        self.p0 = p0
        self.bounds = bounds
        self.param_names = names

        # Sets the default flux scale for finding sensitivity
        # Default value is 1 (Gev)^-1 (cm)^-2 (s)^-1
        self.bkg_ts = None

        # self.clean_true_param_values()

        if cleanup:
            self.clean_pickles()

    def dump_results(self, results, scale, seed):
        """Takes the results of a set of trials, and saves the dictionary as
        a pickle pkl_file. The flux scale is used as a parent directory, and the
        pickle pkl_file itself is saved with a name equal to its random seed.

        :param results: Dictionary of Minimisation results from trials
        :param scale: Scale of inputted flux
        :param seed: Random seed used for running of trials
        """

        write_dir = self.pickle_output_dir + str(float(scale)) + "/"

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

            # if self.flare:
            #     default["Flare Start"] = self.inj_kwargs["Time PDF"].t0

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

    def clean_pickles(self):
        """This function will remove all pre-existing pickle files in the
        output directory, to prevent contamination if the minimisation
        handler arguments are run with different arguments. By default,
        it is not run. The argument cleanup=True must be provided for this
        function to be called. In any case, the user is required to confirm
        the deletion.
        """

        cmd = "rm -rf " + self.pickle_output_dir + "*"

        print "All saved pickle data will be removed, using the following " \
              "command:"
        print
        print "\t", cmd
        print
        print "Is this correct? (y/n)"

        x = ""

        while x not in ["y", "n"]:
            x = raw_input("")

        if x == "y":
            os.system(cmd)

    # def clean_true_param_values(self):
    #     inj_dir = inj_dir_name(self.name)
    #     names = os.listdir(inj_dir)
    #
    #     for name in names:
    #         path = inj_dir + name
    #         os.remove(path)


    def iterate_run(self, scale=1, n_steps=5, n_trials=50):

        scale_range = np.linspace(0., scale, n_steps)

        # truth_dict = dict()

        for scale in scale_range:
            self.run(n_trials, scale)
            # truth_dict[scale] = new

    def run_stacked(self, n_trials=n_trials_default, scale=1):

        seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        param_vals = [[] for x in self.p0]
        ts_vals = []
        flags = []

        print "Generating", n_trials, "trials!"

        for i in range(int(n_trials)):
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

                w = acc * (self.sources["Distance"] ** -2) * np.array(
                    time_weights)

                w = w[:, np.newaxis]

                for j, ind_w in enumerate(w):
                    weights_matrix[i][j] = ind_w

            weights_matrix /= np.sum(weights_matrix)

            # for i, row in enumerate(weights_matrix):
            #     print "Season", i, np.sum(row)
            #
            # raw_input("prompt")

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, (name, f) in enumerate(llh_functions.iteritems()):
                w = weights_matrix[i][:, np.newaxis]
                ts_val += f(params, w)

            return -ts_val

        return f_final

    def run_flare(self, n_trials=n_trials_default, scale=1):

        seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        print "Running", n_trials, "trials"

        results = {
            "TS": []
        }

        for source in self.sources:
            results[source["Name"]] = {
                "TS": [],
                "Parameters": [[] for x in self.param_names]
            }

        for i in range(int(n_trials)):

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    cfg = parser.parse_args()

    print cfg.file

    with open(cfg.file) as f:
        mh_dict = Pickle.load(f)

    mh = MinimisationHandler(mh_dict)
    mh.iterate_run(mh_dict["scale"], mh_dict["n_steps"])

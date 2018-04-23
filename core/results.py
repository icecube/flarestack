import os
import cPickle as Pickle
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import name_pickle_output_dir, plot_output_dir, k_to_flux, fit_setup
from ts_distributions import plot_background_ts_distribution, plot_fit_results


class ResultsHandler:

    def __init__(self, name, llh_kwargs, sources, cleanup=False):
        self.name = name
        self.results = dict()
        self.pickle_output_dir = name_pickle_output_dir(name)
        self.plot_dir = plot_output_dir(name)
        self.merged_dir = self.pickle_output_dir + "merged/"

        p0, bounds, names = fit_setup(llh_kwargs, sources)
        self.names = names
        self.bounds = bounds
        self.p0 = p0

        if cleanup:
            self.clean_merged_data()

        self.sensitivity = np.nan
        self.extrapolated = False

        self.merge_pickle_data()
        self.find_sensitivity()

    def clean_merged_data(self):
        for f in os.listdir(self.merged_dir):
            os.remove(self.merged_dir + f)

    def merge_pickle_data(self):

        all_sub_dirs = [x for x in os.listdir(self.pickle_output_dir)
                        if x[0] != "." and x != "merged"]

        try:
            os.makedirs(self.merged_dir)
        except OSError:
            pass

        for sub_dir_name in all_sub_dirs:
            sub_dir = self.pickle_output_dir + sub_dir_name + "/"

            files = os.listdir(sub_dir)

            merged_path = self.merged_dir + sub_dir_name + ".pkl"

            if os.path.isfile(merged_path):
                with open(merged_path) as mp:
                    merged_data = Pickle.load(mp)
            else:
                merged_data = dict()

            for filename in files:
                path = sub_dir + filename
                with open(path) as f:
                    data = Pickle.load(f)

                    for key in data.keys():
                        if key not in merged_data.keys():
                            merged_data[key] = data[key]
                        else:
                            merged_data[key] += data[key]

                os.remove(path)

            with open(merged_path, "wb") as mp:
                Pickle.dump(merged_data, mp)

            if len(merged_data.keys()) > 0:
                self.results[float(sub_dir_name)] = merged_data

        if len(self.results.keys()) == 0:
            raise Exception("No data was found by ResultsHandler object!")

    def find_sensitivity(self):

        try:
            bkg_dict = self.results[0.0]
        except KeyError:
            raise Exception("No key equal to '0.0'")

        bkg_ts = bkg_dict["TS"]
        bkg_median = np.median(bkg_ts)
        x = sorted(self.results.keys())
        y = []

        for scale in x:
            print scale,
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > bkg_median])) / (float(len(ts_array)))
            print "Fraction of overfluctuations is", "{0:.2f}".format(frac)
            y.append(frac)

            ts_path = self.plot_dir + "ts_distributions/" + str(scale) + ".pdf"
            plot_background_ts_distribution(ts_array, ts_path)

            param_path = self.plot_dir + "params/" + str(scale) + ".pdf"
            plot_fit_results(self.results[scale]["Parameters"], param_path,
                             self.names)

        x = k_to_flux(np.array(x))

        threshold = 0.9

        b = (1 - min(y))

        def f(x, a):
            value = (1 - b * np.exp(-a * x))
            return value

        best_a = scipy.optimize.curve_fit(
            f, x, y, p0=[1e10])[0][0]

        def best_f(x):
            return f(x, best_a)

        self.sensitivity = (1./best_a) * np.log(b / (1 - threshold))

        if self.sensitivity > max(x):
            self.extrapolated = True

        xrange = np.linspace(0.0, 1.1 * max(x), 1000)

        savepath = self.plot_dir + "sensitivity.pdf"

        plt.figure()
        plt.scatter(x, y, color="black")
        plt.plot(xrange, best_f(xrange), color="blue")
        plt.axhline(threshold, lw=1, color="red", linestyle="--")
        plt.axvline(self.sensitivity, lw=2, color="red")
        plt.ylim(0., 1.)
        plt.xlim(0., max(xrange))
        plt.ylabel(r'Overfluctutations relative to median $\lambda_{bkg}$')
        plt.xlabel(r"Flux strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
        plt.savefig(savepath)
        plt.close()

        if self.extrapolated:
            print "EXTRAPOLATED",

        print "Sensitivity is", "{0:.3g}".format(self.sensitivity)

        #

        # self.bkg_ts = ts_array







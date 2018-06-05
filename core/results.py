import os
import cPickle as Pickle
import numpy as np
import math
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import name_pickle_output_dir, plot_output_dir, k_to_flux, \
    fit_setup, inj_dir_name
from ts_distributions import plot_background_ts_distribution, plot_fit_results


class ResultsHandler:

    def __init__(self, name, llh_kwargs, cat_path, show_inj=False,
                 cleanup=False):

        sources = np.load(cat_path)

        self.name = name

        print name

        self.results = dict()
        self.pickle_output_dir = name_pickle_output_dir(name)
        self.plot_dir = plot_output_dir(name)
        self.merged_dir = self.pickle_output_dir + "merged/"

        # Checks if the code should search for flares. By default, this is
        # not done.
        try:
            self.flare = llh_kwargs["Flare Search?"]
        except KeyError:
            self.flare = False

        if self.flare:
            self.make_plots = self.flare_plots
        else:
            self.make_plots = self.noflare_plots

        try:
            self.negative_n_s = llh_kwargs["Fit Negative n_s?"]
        except KeyError:
            self.negative_n_s = False
        #
        # print "negative_ns", self.negative_n_s

        p0, bounds, names = fit_setup(llh_kwargs, sources, self.flare)
        self.param_names = names
        self.bounds = bounds
        self.p0 = p0

        if cleanup:
            self.clean_merged_data()

        self.sensitivity = np.nan
        self.disc_potential = np.nan
        self.extrapolated_sens = False
        self.extrapolated_disc = False
        self.show_inj = show_inj

        if self.show_inj:
            self.inj = self.load_injection_values()
        else:
            self.inj = None

        self.merge_pickle_data()

        self.find_sensitivity()
        self.find_disc_potential()

    def clean_merged_data(self):
        try:
            for f in os.listdir(self.merged_dir):
                os.remove(self.merged_dir + f)
        except OSError:
            pass

    def load_injection_values(self):

        load_dir = inj_dir_name(self.name)

        inj_values = dict()

        for file in os.listdir(load_dir):
            path = load_dir + file

            with open(path) as f:
                inj_values[os.path.splitext(file)[0]] = Pickle.load(f)

        return inj_values


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
                merged_data = {}

            for filename in files:
                path = sub_dir + filename

                with open(path) as f:
                    data = Pickle.load(f)
                os.remove(path)

                for key in ["TS"]:
                    try:
                        merged_data[key] += data[key]
                    except KeyError:
                        merged_data[key] = data[key]

                if "Parameters" in data.keys():

                    if "Paramters" not in merged_data.keys():
                        merged_data["Parameters"] = [
                            [] for x in data["Parameters"]]

                    for k, val in enumerate(data["Parameters"]):
                        try:
                            merged_data["Parameters"][k] += val
                        except IndexError:
                            merged_data["Parameters"][k] = val

                else:
                    keys = [x for x in data.keys() if x not in ["TS"]]

                    if keys[0] not in merged_data.keys():
                        merged_data = data

                    else:
                        for key in keys:
                            merged_data[key]["TS"] += data[key]["TS"]
                            for k, val in enumerate(data[key]["Parameters"]):
                                merged_data[key]["Parameters"][k] += val

                    # for key in data.keys():
                    #     if key not in merged_data.keys():
                    #
                    #     elif isinstance(merged_data[key], dict):
                    #         for subkey in merged_data[key].keys():
                    #             merged_data[key][subkey] += data[key][subkey]
                    #     else:
                    #         merged_data[key] += data[key]

            with open(merged_path, "wb") as mp:
                Pickle.dump(merged_data, mp)

            if len(merged_data.keys()) > 0:
                self.results[float(sub_dir_name)] = merged_data

        if len(self.results.keys()) == 0:
            print "No data was found by ResultsHandler object!"
            return

    def find_sensitivity(self):

        try:
            bkg_dict = self.results[0.0]
        except KeyError:
            print "No key equal to '0.0'"
            return

        bkg_ts = bkg_dict["TS"]

        bkg_median = np.median(bkg_ts)
        x = sorted(self.results.keys())
        x_acc = []
        y = []

        for scale in x:
            print scale,
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > bkg_median])) / (float(len(ts_array)))
            print "Fraction of overfluctuations is", "{0:.2f}".format(frac),
            print "(", len(ts_array), ")"

            if len(ts_array) > 1:
                y.append(frac)
                x_acc.append(scale)

                self.make_plots(scale)

        x = np.array(x_acc)

        x_flux = k_to_flux(x)

        threshold = 0.9

        b = (1 - min(y))

        def f(x, a):
            value = (1 - b * np.exp(-a * x))
            return value

        best_a = scipy.optimize.curve_fit(
            f, x, y,  p0=[0.1])[0][0]

        # print "best_a", best_a

        def best_f(x):
            return f(x, best_a)

        self.sensitivity = k_to_flux((1./best_a) * np.log(b / (1 - threshold)))

        if self.sensitivity > max(x_flux):
            self.extrapolated_sens = True

        xrange = np.linspace(0.0, 1.1 * max(x), 1000)

        savepath = self.plot_dir + "sensitivity.pdf"

        plt.figure()
        plt.scatter(x_flux, y, color="black")
        plt.plot(k_to_flux(xrange), best_f(xrange), color="blue")
        plt.axhline(threshold, lw=1, color="red", linestyle="--")
        plt.axvline(self.sensitivity, lw=2, color="red")
        plt.ylim(0., 1.)
        plt.xlim(0., k_to_flux(max(xrange)))
        plt.ylabel(r'Overfluctuations relative to median $\lambda_{bkg}$')
        plt.xlabel(r"Flux strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
        plt.savefig(savepath)
        plt.close()

        if self.extrapolated_sens:
            print "EXTRAPOLATED",

        print "Sensitivity is", "{0:.3g}".format(self.sensitivity)

    def find_disc_potential(self):

        ts_path = self.plot_dir + "ts_distributions/0.0.pdf"

        try:
            bkg_dict = self.results[0.0]
        except KeyError:
            print "No key equal to '0.0'"
            return

        bkg_ts = bkg_dict["TS"]

        disc_threshold = plot_background_ts_distribution(bkg_ts, ts_path,
                                                         self.negative_n_s)
        bkg_median = np.median(bkg_ts)
        x = sorted(self.results.keys())
        y = []

        for scale in x:
            print scale,
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > disc_threshold])) / (
                float(len(ts_array)))
            print "Fraction of overfluctuations is", "{0:.2f}".format(frac)
            y.append(frac)

        x = np.array(x)

        x_flux = k_to_flux(x)

        threshold = 0.5

        def f(x, a, b):
            value = 0.5 * (np.tanh(a*x - b) + 1.)
            return value

        [best_a, best_b] = scipy.optimize.curve_fit(
            f, x, y,  p0=[0.001, 0.])[0]

        def best_f(x):
            return f(x, best_a, best_b)

        sol = best_b/best_a

        self.disc_potential = k_to_flux(sol)

        if self.disc_potential > max(x_flux):
            self.extrapolated_disc = True

        xrange = np.linspace(0.0, 1.1 * max(x), 1000)

        savepath = self.plot_dir + "disc.pdf"

        plt.figure()
        plt.scatter(x_flux, y, color="black")
        plt.plot(k_to_flux(xrange), best_f(xrange), color="blue")
        plt.axhline(threshold, lw=1, color="red", linestyle="--")
        plt.axvline(self.sensitivity, lw=2, color="black", linestyle="--")
        plt.axvline(self.disc_potential, lw=2, color="red")
        plt.ylim(0., 1.)
        plt.xlim(0., k_to_flux(max(xrange)))
        plt.ylabel(r'Overfluctuations relative to 90\% of $\lambda_{bkg}$')
        plt.xlabel(r"Flux [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
        plt.savefig(savepath)
        plt.close()

        if self.extrapolated_disc:
            print "EXTRAPOLATED",

        print "Discovery Potential is", "{0:.3g}".format(self.disc_potential)

    def noflare_plots(self, scale):
        ts_array = np.array(self.results[scale]["TS"])
        ts_path = self.plot_dir + "ts_distributions/" + str(scale) + ".pdf"
        plot_background_ts_distribution(ts_array, ts_path, self.negative_n_s)

        param_path = self.plot_dir + "params/" + str(scale) + ".pdf"

        if self.show_inj:
            inj = self.inj[str(scale)]
        else:
            inj = None

        plot_fit_results(self.results[scale]["Parameters"], param_path,
                         self.param_names, inj=inj)

    def flare_plots(self, scale):
        ts_array = np.array(self.results[scale]["TS"])
        ts_path = self.plot_dir + "ts_distributions/" + str(scale) + ".pdf"

        sources = [x for x in self.results[scale].keys() if x != "TS"]

        for source in sources:
            ts_array = np.array(self.results[scale][source]["TS"])
            ts_path = self.plot_dir + source + "/ts_distributions/" + str(
                scale) + ".pdf"
            plot_background_ts_distribution(ts_array, ts_path)

            param_path = self.plot_dir + source + "/params/" + str(scale) + \
                         ".pdf"

            if self.show_inj:
                for key in self.inj.keys():
                    if '{0:.4G}'.format(float(key)) == '{0:.4G}'.format(
                            float(scale)):
                        inj = self.inj[key]
                        print inj
                        raw_input("prompt")
            else:
                inj = None

            plot_fit_results(self.results[scale][source]["Parameters"],
                             param_path, self.param_names,
                             inj=inj)







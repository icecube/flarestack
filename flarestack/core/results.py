import os
import cPickle as Pickle
import numpy as np
import math
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from flarestack.shared import name_pickle_output_dir, plot_output_dir, k_to_flux, \
    fit_setup, inj_dir_name, scale_shortener
from flarestack.core.ts_distributions import plot_background_ts_distribution, \
    plot_fit_results
from flarestack.utils.neutrino_astronomy import calculate_astronomy


class ResultsHandler:

    def __init__(self, name, llh_kwargs, cat_path, show_inj=True,
                 cleanup=False):

        self.sources = np.sort(np.load(cat_path), order="Distance (Mpc)")

        self.name = name

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

        # Checks whether negative n_s is fit or not

        try:
            self.negative_n_s = llh_kwargs["Fit Negative n_s?"]
        except KeyError:
            self.negative_n_s = False

        try:
            self.fit_weights = llh_kwargs["Fit Weights?"]
        except KeyError:
            self.fit_weights = False

        # Sets default Chi2 distribution to fit to background trials

        if self.fit_weights:
            self.ts_type = "Fit Weights"
        elif self.flare:
            self.ts_type = "Flare"
        elif self.negative_n_s:
            self.ts_type = "Negative n_s"
        else:
            self.ts_type = "Standard"
        #
        # print "negative_ns", self.negative_n_s

        p0, bounds, names = fit_setup(llh_kwargs, self.sources, self.flare)
        self.param_names = names
        self.bounds = bounds
        self.p0 = p0

        if cleanup:
            self.clean_merged_data()

        self.sensitivity = np.nan
        self.bkg_median = np.nan
        self.frac_over = np.nan
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

        try:
            self.find_disc_potential()
        except RuntimeError:
            pass

    def astro_values(self, e_pdf_dict):
        """Function to convert the values calculated for sensitivity and
        discovery potential, which are given in terms of flux at the
        detector, to physical quantities for a source of mean luminosity. The
        fluxes are integrated over an energy range, either specified in
        e_pdf_dict, or by default between 100GeV and 10PeV. They are then
        scaled by the luminosity distance to source, giving the mean
        luminosity of the sources in the catalogue. The assumption is that
        the sources are standard candles, so this value would be the same for
        each source, and is thus only calculated once. To convert further from
        this mean luminosity to the luminosity of a specific source,
        the values must be multiplied by the "relative injection weight" of
        the source, which has a mean of 1.

        :param e_pdf_dict: Dictionary containing energy PDF information
        :return: Values for the neutrino luminosity sensitivity and
        discovery potential
        """

        astro_sens = calculate_astronomy(self.sensitivity, e_pdf_dict,
                                         self.sources)
        astro_disc = calculate_astronomy(self.disc_potential, e_pdf_dict,
                                         self.sources)

        return astro_sens, astro_disc

    def clean_merged_data(self):
        """Function to clear cache of all data"""
        try:
            for f in os.listdir(self.merged_dir):
                os.remove(self.merged_dir + f)
        except OSError:
            pass

    def load_injection_values(self):
        """Function to load the values used in injection, so that a
        comparison to the fit results can be made.

        :return: Dictionary of injected values.
        """

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

                try:
                    with open(path) as f:
                        data = Pickle.load(f)
                except EOFError:
                    print "Failed loading", path
                    continue
                os.remove(path)

                for key in ["TS"]:
                    try:
                        merged_data[key] += data[key]
                    except KeyError:
                        merged_data[key] = data[key]

                if "Parameters" in data.keys():

                    if "Parameters" not in merged_data.keys():
                        merged_data["Parameters"] = [
                            [] for _ in data["Parameters"]]

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
                self.results[scale_shortener(float(sub_dir_name))] = merged_data

        if len(self.results.keys()) == 0:
            print "No data was found by ResultsHandler object!"
            return

    def find_sensitivity(self):
        """Uses the results of the background trials to find the median TS
        value, determining the sensitivity threshold. This sensitivity is
        not necessarily zero, for example with negative n_s, fitting of
        weights or the flare search method. Uses the values of
        injection trials to fit an 1-exponential decay function to the
        overfluctuations, allowing for calculation of the sensitivity.
        Where the injected flux was not sufficient to reach the
        sensitivity, extrapolation will be used instead of interpolation,
        but this will obviously have larger associated errors. If
        extrapolation is used, self.extrapolated_sens is set to true. In
        either case, a plot of the overfluctuations as a function of the
        injected signal will be made.
        """

        try:
            bkg_dict = self.results[scale_shortener(0.0)]
        except KeyError:
            print "No key equal to '0'"
            return

        bkg_ts = bkg_dict["TS"]

        bkg_median = np.median(bkg_ts)
        self.bkg_median = bkg_median
        x = sorted(self.results.keys())
        x_acc = []
        y = []

        x = [scale_shortener(i) for i in sorted([float(j) for j in x])]

        for scale in x:
            print scale,
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > bkg_median])) / (float(len(
                ts_array)))
            print "Fraction of overfluctuations is", "{0:.2f}".format(frac),
            print "above", "{0:.2f}".format(bkg_median),
            print "(", len(ts_array), ")"

            if scale == scale_shortener(0.0):
                self.frac_over = frac

            if len(ts_array) > 1:
                y.append(frac)
                x_acc.append(float(scale))

                self.make_plots(scale)

            # raw_input("prompt")

        x = np.array(x_acc)

        x_flux = k_to_flux(x)

        threshold = 0.9

        b = (1 - min(y))

        def f(x, a):
            value = (1 - b * np.exp(-a * x))
            return value

        best_a = scipy.optimize.curve_fit(
            f, x, y,  p0=[1./max(x)])[0][0]

        # print "best_a", best_a
        # raw_input("prompt")

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

        ts_path = self.plot_dir + "ts_distributions/0.pdf"

        try:
            bkg_dict = self.results[scale_shortener(0.0)]
        except KeyError:
            print "No key equal to '0'"
            return

        bkg_ts = bkg_dict["TS"]

        disc_threshold = plot_background_ts_distribution(
            bkg_ts, ts_path, ts_type=self.ts_type)
        bkg_median = np.median(bkg_ts)
        x = sorted(self.results.keys())
        y = []

        x = [scale_shortener(i) for i in sorted([float(j) for j in x])]

        for scale in x:
            print scale,
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > disc_threshold])) / (
                float(len(ts_array)))
            print "Fraction of overfluctuations is", "{0:.2f}".format(frac)
            y.append(frac)

        x = np.array([float(s) for s in x])

        x_flux = k_to_flux(x)

        threshold = 0.5

        def f(x, a, b, c):
            value = scipy.stats.gamma.cdf(x, a, b, c)
            return value

        res = scipy.optimize.curve_fit(
            f, x, y,  p0=[6, -0.1 * max(x), 0.1 * max(x)])

        best_a = res[0][0]
        best_b = res[0][1]
        best_c = res[0][2]

        def best_f(x):
            return f(x, best_a, best_b, best_c)

        sol = scipy.stats.gamma.ppf(0.5, best_a, best_b, best_c)

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
        plt.ylabel(r'Overfluctuations relative to 5 $\sigma$ Threshold')
        plt.xlabel(r"Flux [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
        plt.savefig(savepath)
        plt.close()

        if self.extrapolated_disc:
            print "EXTRAPOLATED",

        print "Discovery Potential is", "{0:.3g}".format(self.disc_potential)

    def noflare_plots(self, scale):
        ts_array = np.array(self.results[scale]["TS"])
        ts_path = self.plot_dir + "ts_distributions/" + str(scale) + ".pdf"

        plot_background_ts_distribution(ts_array, ts_path,
                                        ts_type=self.ts_type)

        param_path = self.plot_dir + "params/" + str(scale) + ".pdf"

        if self.show_inj:
            inj = self.inj[str(scale)]
        else:
            inj = None

        plot_fit_results(self.results[scale]["Parameters"], param_path,
                         self.param_names, inj=inj)

    def flare_plots(self, scale):

        sources = [x for x in self.results[scale].keys() if x != "TS"]

        for source in sources:

            ts_array = np.array(self.results[scale][source]["TS"])
            ts_path = self.plot_dir + source + "/ts_distributions/" + str(
                scale) + ".pdf"

            plot_background_ts_distribution(ts_array, ts_path,
                                            ts_type=self.ts_type)

            param_path = self.plot_dir + source + "/params/" + str(scale) + \
                         ".pdf"

            if self.show_inj:
                inj = self.inj[str(scale)]
            else:
                inj = None

            plot_fit_results(self.results[scale][source]["Parameters"],
                             param_path, self.param_names,
                             inj=inj)







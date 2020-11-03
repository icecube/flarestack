import os
import pickle as Pickle
import numpy as np
import scipy
import scipy.stats
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from flarestack.shared import name_pickle_output_dir, plot_output_dir, \
    k_to_flux, inj_dir_name, scale_shortener, flux_to_k
from flarestack.core.ts_distributions import plot_background_ts_distribution, \
    plot_fit_results
from flarestack.utils.neutrino_astronomy import calculate_astronomy
from flarestack.core.minimisation import MinimisationHandler
from flarestack.utils.catalogue_loader import load_catalogue
import sys
import logging

logger = logging.getLogger(__name__)

class OverfluctuationError(Exception):
    pass

class ResultsHandler(object):

    def __init__(self, rh_dict):

        self.sources = load_catalogue(rh_dict["catalogue"])

        self.name = rh_dict["name"]
        self.mh_name = rh_dict['mh_name']

        self.results = dict()
        self.pickle_output_dir = name_pickle_output_dir(self.name)
        self.plot_dir = plot_output_dir(self.name)
        self.merged_dir = os.path.join(self.pickle_output_dir, "merged")

        # Checks if the code should search for flares. By default, this is
        # not done.
        # try:
        #     self.flare = llh_kwargs["Flare Search?"]
        # except KeyError:
        #     self.flare = False

        # if self.flare:
        #     self.make_plots = self.flare_plots
        # else:
        self.make_plots = self.noflare_plots

        # Checks whether negative n_s is fit or not
        #
        # try:
        #     self.negative_n_s = llh_kwargs["Fit Negative n_s?"]
        # except KeyError:
        #     self.negative_n_s = False
        #
        # try:
        #     self.fit_weights = llh_kwargs["Fit Weights?"]
        # except KeyError:
        #     self.fit_weights = False

        # Sets default Chi2 distribution to fit to background trials
        #
        # if self.fit_weights:
        #     self.ts_type = "Fit Weights"
        # elif self.flare:
        #     self.ts_type = "Flare"
        # elif self.negative_n_s:
        #     self.ts_type = "Negative n_s"
        # else:
        self.ts_type = "Standard"
        #
        # print "negative_ns", self.negative_n_s

        p0, bounds, names = MinimisationHandler.find_parameter_info(rh_dict)

        # p0, bounds, names = fit_setup(llh_kwargs, self.sources, self.flare)
        self.param_names = names
        self.bounds = bounds
        self.p0 = p0

        # if cleanup:
        #     self.clean_merged_data()

        self.sensitivity = np.nan
        self.sensitivity_err = np.nan
        self.bkg_median = np.nan
        self.frac_over = np.nan
        self.disc_potential = np.nan
        self.disc_err = np.nan
        self.disc_potential_25 = np.nan
        self.disc_ts_threshold = np.nan
        self.extrapolated_sens = False
        self.extrapolated_disc = False
        self.flux_to_ns = np.nan

        # if self.show_inj:
        self.inj = self.load_injection_values()
        # else:
        #     self.inj = None

        try:
            self.merge_pickle_data()
        except FileNotFoundError:
            logger.warning("No files found at {0}".format(self.pickle_output_dir))

        try:
            self.find_ns_scale()
        except ValueError as e:
            logger.warning("RuntimeError for ns scale factor: \n {0}".format(e))

        self.plot_bias()

        try:
            self.find_sensitivity()
        except ValueError as e:
            logger.warning("RuntimeError for discovery potential: \n {0}".format(e))

        try:
            self.find_disc_potential()
        except RuntimeError as e:
            logger.warning("RuntimeError for discovery potential: \n {0}".format(e))
        except TypeError as e:
            logger.warning("TypeError for discovery potential: \n {0}".format(e))
        except ValueError as e:
            logger.warning("TypeError for discovery potential: \n {0}".format(e))


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

        astro_sens = self.nu_astronomy(self.sensitivity, e_pdf_dict)
        astro_disc = self.nu_astronomy(self.disc_potential, e_pdf_dict)

        return astro_sens, astro_disc


    def nu_astronomy(self, flux, e_pdf_dict):
        """Function to convert a local flux in the detector at 1GeV to physical
        quantities for a source of mean luminosity. The
        fluxes are integrated over an energy range, either specified in
        e_pdf_dict, or by default between 100GeV and 10PeV. They are then
        scaled by the luminosity distance to source, giving the mean
        luminosity of the sources in the catalogue. The assumption is that
        the sources are standard candles, so this value would be the same for
        each source, and is thus only calculated once. To convert further from
        this mean luminosity to the luminosity of a specific source,
        the values must be multiplied by the "relative injection weight" of
        the source, which has a mean of 1.

        :param flux: Flux to be converted
        :param e_pdf_dict: Dictionary containing energy PDF information
        :return: Value for the neutrino luminosity
        """
        return calculate_astronomy(flux, e_pdf_dict, self.sources)

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
            path = os.path.join(load_dir, file)

            with open(path, "rb") as f:
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
            sub_dir = os.path.join(self.pickle_output_dir,  sub_dir_name)

            files = os.listdir(sub_dir)

            merged_path = os.path.join(self.merged_dir, sub_dir_name + ".pkl")

            if os.path.isfile(merged_path):
                with open(merged_path, "rb") as mp:
                    merged_data = Pickle.load(mp)
            else:
                merged_data = {}

            for filename in files:
                path = os.path.join(sub_dir, filename)

                try:
                    with open(path, "rb") as f:
                        data = Pickle.load(f)
                except EOFError:
                    logger.warning("Failed loading: {0}".format(path))
                    continue
                os.remove(path)

                if merged_data == {}:
                    merged_data = data
                else:
                    for (key, info) in data.items():
                        if isinstance(info, list):
                            merged_data[key] += info
                        else:
                            for (param_name, params) in info.items():
                                try: merged_data[key][param_name] += params
                                except KeyError as m:
                                    logger.warning('Keys [{key}][{param_name}] not found in \n {merged_data}')
                                    raise KeyError(m)

            with open(merged_path, "wb") as mp:
                Pickle.dump(merged_data, mp)

            if len(list(merged_data.keys())) > 0:
                self.results[scale_shortener(float(sub_dir_name))] = merged_data

        if len(list(self.results.keys())) == 0:
            logger.warning("No data was found by ResultsHandler object! \n")
            logger.warning("Tried root directory: \n {0} \n ".format(self.pickle_output_dir))
            sys.exit()

    def find_ns_scale(self):
        """Find the number of neturinos corresponding to flux
        """
        x = sorted([float(x) for x in self.results.keys()])
        raw_x = [scale_shortener(i) for i in sorted([float(j) for j in x])]

        try:
            # if weights were not fitted, number of neutrinos is stored in just one parameter
            if not 'fit' in self.mh_name:
                self.flux_to_ns = self.inj[raw_x[1]]["n_s"] / k_to_flux(float(x[1]))

            # if weights were fitted, there is one n_s for each fitted source
            else:
                sc_dict = self.inj[raw_x[1]]
                self.flux_to_ns = sum([sc_dict[k] for k in sc_dict if 'n_s' in str(k)]) / k_to_flux(float(x[1]))

            logger.debug(f"Conversion ratio of flux to n_s: {self.flux_to_ns:.2f}")

        except KeyError:
            logger.warning(f"KeyError: key \"n_s\" not found and minimizer is {self.mh_name}!!")

    def find_sensitivity(self):
        """Uses the results of the background trials to find the median TS
        value, determining the sensitivity threshold. This sensitivity is
        not necessarily zero, for example with negative n_s, fitting of
        weights or the flare search method.
        """

        try:
            bkg_dict = self.results[scale_shortener(0.0)]
        except KeyError:
            logger.error("No key equal to '0'")
            return

        bkg_ts = bkg_dict["TS"]

        bkg_median = np.median(bkg_ts)
        self.bkg_median = bkg_median

        savepath = os.path.join(self.plot_dir, "sensitivity.pdf")

        self.sensitivity, self.extrapolated_sens, self.sensitivity_err = self.find_overfluctuations(
            bkg_median, savepath)

        msg = ""

        if self.extrapolated_sens:
            msg = "EXTRAPOLATED "

        logger.info("{0}Sensitivity is {1:.3g}".format(msg, self.sensitivity))

    # def set_upper_limit(self, ts_val, savepath):
    #     """Set an upper limit, based on a Test Statistic value from
    #     unblinding, as well as a
    #
    #     :param ts_val: Test Statistic Value
    #     :param savepath: Path to save plot
    #     :return: Upper limit, and whether this was extrapolated
    #     """
    #
    #     try:
    #         bkg_dict = self.results[scale_shortener(0.0)]
    #     except KeyError:
    #         print "No key equal to '0'"
    #         return
    #
    #     bkg_ts = bkg_dict["TS"]
    #     bkg_median = np.median(bkg_ts)
    #
    #     # Set an upper limit based on the Test Statistic value for an
    #     # overfluctuation, or the median background for an underfluctuation.
    #
    #     ref_ts = max(ts_val, bkg_median)
    #
    #     ul, extrapolated = self.find_overfluctuations(
    #         ref_ts, savepath)
    #
    #     if extrapolated:
    #         print "EXTRAPOLATED",
    #
    #     print "Upper limit is", "{0:.3g}".format(ul)
    #     return ul, extrapolated

    def find_overfluctuations(self, ts_val, savepath):
        """Uses the values of injection trials to fit an 1-exponential decay
        function to the overfluctuations, allowing for calculation of the
        sensitivity. Where the injected flux was not sufficient to reach the
        sensitivity, extrapolation will be used instead of interpolation,
        but this will obviously have larger associated errors. If
        extrapolation is used, self.extrapolated_sens is set to true. In
        either case, a plot of the overfluctuations as a function of the
        injected signal will be made.
        """

        x = sorted(self.results.keys())
        x_acc = []
        y = []

        x = [scale_shortener(i) for i in sorted([float(j) for j in x])]

        yerr = []

        for scale in x:
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > ts_val])) / (float(len(
                ts_array)))
            
            logger.info(
                "Fraction of overfluctuations is {0:.2f} above {1:.2f} (N_trials={2}) (Scale={3})".format(
                    frac, ts_val, len(ts_array), scale
                )
            )

            if scale == scale_shortener(0.0):
                self.frac_over = frac

            if len(ts_array) > 1:
                y.append(frac)
                x_acc.append(float(scale))
                yerr.append(1./np.sqrt(float(len(ts_array))))

                self.make_plots(scale)

        x = np.array(x_acc)

        x_flux = k_to_flux(x)

        threshold = 0.9

        b = (1 - min(y))

        def f(x, a):
            value = (1 - b * np.exp(-a * x))
            return value

        popt, pcov = scipy.optimize.curve_fit(
            f, x, y,  sigma=yerr, absolute_sigma=True, p0=[1./max(x)])

        perr = np.sqrt(np.diag(pcov))

        best_a = popt[0]

        def best_f(x, sd=0.):
            a = best_a + perr*sd
            return f(x, a)

        fit = k_to_flux((1./best_a) * np.log(b / (1 - threshold)))

        if fit > max(x_flux):
            logger.warning("The sensitivity is beyond the range of the tested scales."
                            "The number is probably not good.")
            extrapolated = True
        else:
            extrapolated = False

        xrange = np.linspace(0.0, 1.1 * max(x), 1000)

        lower = k_to_flux((1./(best_a + perr)) * np.log(b / (1 - threshold)))
        upper = k_to_flux((1./(best_a - perr)) * np.log(b / (1 - threshold)))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.errorbar(x_flux, y, yerr=yerr, color="black", fmt=" ", marker="o")
        ax1.plot(k_to_flux(xrange), best_f(xrange), color="blue")
        ax1.fill_between(k_to_flux(xrange), best_f(xrange, 1), best_f(xrange, -1), color="blue", alpha=0.1)
        ax1.axhline(threshold, lw=1, color="red", linestyle="--")
        ax1.axvline(fit, lw=2, color="red")
        ax1.axvline(lower, lw=2, color="red", linestyle=":")
        ax1.axvline(upper, lw=2, color="red", linestyle=":")
        ax1.set_ylim(0., 1.)
        ax1.set_xlim(0., k_to_flux(max(xrange)))
        ax1.set_ylabel('Overfluctuations above TS=' + "{:.2f}".format(ts_val))
        plt.xlabel(r"Flux Normalisation @ 1GeV [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")

        if not np.isnan(self.flux_to_ns):
            ax2 = ax1.twiny()
            ax2.grid(0)
            ax2.set_xlim(0., self.flux_to_ns * k_to_flux(max(xrange)))
            ax2.set_xlabel(r"Number of neutrinos")

        fig.savefig(savepath)
        plt.close()

        if len(np.where(np.array(y) < 0.95)[0]) < 2:
            raise OverfluctuationError(f"Not enough points with overfluctuations under 95%, lower injection scale!")

        sens_err = np.array([fit - lower, upper - fit]).T[0]

        return fit, extrapolated, sens_err

    def find_disc_potential(self):

        ts_path = os.path.join(self.plot_dir, "ts_distributions/0.pdf")

        try:
            bkg_dict = self.results[scale_shortener(0.0)]
        except KeyError:
            logger.error("No key equal to '0'")
            return

        bkg_ts = bkg_dict["TS"]

        disc_threshold = plot_background_ts_distribution(
            bkg_ts, ts_path, ts_type=self.ts_type)

        self.disc_ts_threshold = disc_threshold

        bkg_median = np.median(bkg_ts)
        x = sorted(self.results.keys())
        y = []
        y_25 = []

        x = [scale_shortener(i) for i in sorted([float(j) for j in x])]

        for scale in x:
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > disc_threshold])) / (
                float(len(ts_array)))

            logger.info(
                "Fraction of overfluctuations is {0:.2f} above {1:.2f} (N_trials={2}) (Scale={3})".format(
                    frac, disc_threshold, len(ts_array), scale
                )
            )

            y.append(frac)
            frac_25 = float(len(ts_array[ts_array > 25.])) / (
                float(len(ts_array)))

            logger.info(
                "Fraction of overfluctuations is {0:.2f} above 25 (N_trials={1}) (Scale={2})".format(
                    frac_25, len(ts_array), scale
                )
            )

            y_25.append(frac_25)

        x = np.array([float(s) for s in x])

        x_flux = k_to_flux(x)

        threshold = 0.5

        sols = []

        for i, y_val in enumerate([y, y_25]):

            def f(x, a, b, c):
                value = scipy.stats.gamma.cdf(x, a, b, c)
                return value

            res = scipy.optimize.curve_fit(
                f, x, y_val,  p0=[6, -0.1 * max(x), 0.1 * max(x)])

            best_a = res[0][0]
            best_b = res[0][1]
            best_c = res[0][2]

            def best_f(x):
                return f(x, best_a, best_b, best_c)

            sol = scipy.stats.gamma.ppf(0.5, best_a, best_b, best_c)
            setattr(self, ["disc_potential", "disc_potential_25"][i],
                    k_to_flux(sol))

            xrange = np.linspace(0.0, 1.1 * max(x), 1000)

            savepath = os.path.join(self.plot_dir, "disc" + ["", "_25"][i] + ".pdf")

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_flux, y_val, color="black")
            ax1.plot(k_to_flux(xrange), best_f(xrange), color="blue")
            ax1.axhline(threshold, lw=1, color="red", linestyle="--")
            ax1.axvline(self.sensitivity, lw=2, color="black", linestyle="--")
            ax1.axvline(self.disc_potential, lw=2, color="red")
            ax1.set_ylim(0., 1.)
            ax1.set_xlim(0., k_to_flux(max(xrange)))
            ax1.set_ylabel(r'Overfluctuations relative to 5 $\sigma$ Threshold')
            plt.xlabel(r"Flux Normalisation @ 1GeV [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")

            if not np.isnan(self.flux_to_ns):
                ax2 = ax1.twiny()
                ax2.grid(0)
                ax2.set_xlim(0., self.flux_to_ns * k_to_flux(max(xrange)))
                ax2.set_xlabel(r"Number of neutrinos")

            fig.savefig(savepath)
            plt.close()

        if self.disc_potential > max(x_flux):
            self.extrapolated_disc = True

        msg = ""

        if self.extrapolated_disc:
            msg = "EXTRAPOLATED "

        logger.info("{0}Discovery Potential is {1:.3g}".format(msg, self.disc_potential))
        logger.info("Discovery Potential (TS=25) is {0:.3g}".format(self.disc_potential_25))

    def noflare_plots(self, scale):
        ts_array = np.array(self.results[scale]["TS"])
        ts_path = os.path.join(self.plot_dir, "ts_distributions/" + str(scale) + ".pdf")

        plot_background_ts_distribution(ts_array, ts_path,
                                        ts_type=self.ts_type)

        param_path = os.path.join(self.plot_dir, "params/" + str(scale) + ".pdf")

        # if self.show_inj:
        inj = self.inj[str(scale)]

        plot_fit_results(self.results[scale]["Parameters"], param_path,
                         inj=inj)

    def ts_evolution_gif(self, n_scale_steps=None, cmap_name='winter'):

        logger.debug('making animation')

        all_scales_list = list(self.results.keys())
        n_scales_all = len(all_scales_list)

        n_scale_steps = n_scales_all - 1 if not n_scale_steps else n_scale_steps

        scale_step_length = int(round(n_scales_all / (n_scale_steps)))
        scales = [all_scales_list[min([i * scale_step_length, n_scales_all - 1])]
                  for i in range(n_scale_steps + 1)]

        ts_arrays = [np.array(self.results[scale]['TS']) for scale in scales]

        ns_arrays = np.array([
            np.array(
                [np.median(self.results[scale]['Parameters'][key])
                 for key in self.results[scale]['Parameters']
                 if 'n_s' in key]
            )
            for scale in scales
        ])

        n_s = [sum(a) for a in ns_arrays]
        logger.debug('numbers of injected neutrinos: ' + str(n_s))

        norm = colors.Normalize(vmin=0, vmax=max(n_s))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        cmap = mappable.get_cmap()

        sq_fig, sq_ax = plt.subplots()
        sq_fig.set_tight_layout(True)
        sq_ax.set_xlim([-5, max(ts_arrays[-1]) + 10])
        sq_ax.set_yscale('log')
        sq_ax.set_xlabel('Test Statistic')
        sq_ax.set_ylabel('a.u.')

        sqbar = sq_fig.colorbar(mappable, ax=sq_ax)
        sqbar.set_label(r'n$_{\mathrm{injected}}$')

        def update(i):
            its = ts_arrays[i]
            ins = n_s[i]
            sq_ax.hist(its, histtype='stepfilled', density=True, color=cmap(ins / max(n_s)))
            sq_ax.set_title(r'n$_{\mathrm{injected}}=$' + '{:.2f}'.format(ins))

        anim = animation.FuncAnimation(
            sq_fig, update, frames=np.arange(0, n_scale_steps), interval=500
        )

        anim_name = os.path.join(self.plot_dir, "ts_distributions/ts_distributions_evolution.gif")
        logger.debug('saving animation under ' + anim_name)
        anim.save(anim_name, dpi=80, writer='imagemagick')

    def ts_distribution_evolution(self):

        logger.debug('plotting evolution of TS distribution')

        all_scales = np.array(list(self.results.keys()))
        all_scales_floats = [float(sc) for sc in all_scales]

        logger.debug('all scales: ' + str(all_scales_floats))
        logger.debug('sensitivity scale: ' + str(flux_to_k(self.sensitivity)))

        sens_scale = all_scales[all_scales_floats >= np.array(flux_to_k(self.sensitivity))][0]
        disc_scale = all_scales[all_scales_floats >= np.array(flux_to_k(self.disc_potential))][0]


        scales = [all_scales[0], sens_scale, disc_scale]
        ts_arrays = [np.array(self.results[scale]['TS']) for scale in scales]
        ns_arrays = np.array([
            np.array(
                    [np.median(self.results[scale]['Parameters'][key])
                     for key in self.results[scale]['Parameters']
                     if 'n_s' in key]
            )
            for scale in scales
        ])

        n_s = [sum(a) for a in ns_arrays]
        logger.debug('numbers of injected neutrinos: ' + str(n_s))

        fig, ax = plt.subplots()

        ax.hist(ts_arrays[0], histtype='stepfilled', label='background', density=True, alpha=0.6, color='blue')


        ax.hist(ts_arrays[1], histtype='step', density=True, color='orange',
                label='signal: {:.2} signal neutrinos'.format(n_s[1]))
        ax.axvline(self.bkg_median, ls='--', label='sensitivity threshold', color='orange')

        ax.hist(ts_arrays[2], histtype='step', density=True, color='red',
                label='signal: {:.2} signal neutrinos'.format(n_s[2]))
        ax.axvline(self.disc_ts_threshold, ls='--', label='discovery potential threshold',
                   color='red')

        ax.set_xlabel('Test Statistic')
        ax.set_ylabel('a.u.')
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()

        sn = os.path.join(self.plot_dir, "ts_distributions/ts_evolution_.pdf")
        logger.debug('saving plot to ' + sn)
        fig.savefig(sn)

        plt.close()

    # def flare_plots(self, scale):
    #
    #     sources = [x for x in self.results[scale].keys() if x != "TS"]
    #
    #     for source in sources:
    #
    #         ts_array = np.array(self.results[scale][source]["TS"])
    #         ts_path = self.plot_dir + source + "/ts_distributions/" + str(
    #             scale) + ".pdf"
    #
    #         plot_background_ts_distribution(ts_array, ts_path,
    #                                         ts_type=self.ts_type)
    #
    #         param_path = self.plot_dir + source + "/params/" + str(scale) + \
    #                      ".pdf"
    #
    #         if self.show_inj:
    #             inj = self.inj[str(scale)]
    #         else:
    #             inj = None
    #
    #         plot_fit_results(self.results[scale][source]["Parameters"],
    #                          param_path, inj)

    def plot_bias(self):
        x = sorted(self.results.keys())
        raw_x = [scale_shortener(i) for i in sorted([float(j) for j in x])]
        base_x = [k_to_flux(float(j)) for j in raw_x]
        base_x_label = r"$\Phi_{1GeV}$ (GeV$^{-1}$ cm$^{-2}$)"

        for i, param in enumerate(self.param_names):

            plt.figure()

            ax = plt.subplot(111)

            meds = []
            ulims = []
            llims = []
            trues = []

            for scale in raw_x:
                vals = self.results[scale]["Parameters"][param]
                med = np.median(vals)
                meds.append(med)
                sig = np.std(vals)
                ulims.append(med + sig)
                llims.append(med - sig)

                true = self.inj[scale][param]
                trues.append(true)

            if "n_s" in param:
                x = trues
                x_label = r"$n_{injected}$" + param.replace("n_s", "")
            else:
                x = base_x
                x_label = base_x_label

            plt.scatter(x, meds, color="orange")
            plt.plot(x, meds, color="black")
            plt.plot(x, trues, linestyle="--", color="red")
            plt.fill_between(x, ulims, llims, alpha=0.5, color="orange")

            ax.set_xlim(left=0.0, right=max(x))
            if min(trues) == 0.0:
                ax.set_ylim(bottom=0.0)

            plt.xlabel(x_label)
            plt.ylabel(param)
            plt.title("Bias (" + param + ")")

            savepath = os.path.join(self.plot_dir, "bias_" + param + ".pdf")
            logger.info("Saving bias plot to {0}".format(savepath))

            try:
                os.makedirs(os.path.dirname(savepath))
            except OSError:
                pass

            plt.savefig(savepath)
            plt.close()










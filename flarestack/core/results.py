import logging
import os
import pickle as Pickle
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.time_pdf import TimePDF
from flarestack.core.ts_distributions import (
    get_ts_fit_type,
    plot_background_ts_distribution,
    plot_fit_results,
)
from flarestack.shared import (
    flux_to_k,
    inj_dir_name,
    k_to_flux,
    name_pickle_output_dir,
    plot_output_dir,
    scale_shortener,
)
from flarestack.utils.catalogue_loader import load_catalogue
from flarestack.utils.neutrino_astronomy import calculate_astronomy

logger = logging.getLogger(__name__)


class OverfluctuationError(Exception):
    pass


class PickleCache:
    def __init__(self, pickle_path: Path, background_only: bool = False):
        self.path = pickle_path
        # self.pickle_path.mkdir(parents=True, exist_ok=True)
        self.merged_path = self.path / "merged"
        self.merged_path.mkdir(parents=True, exist_ok=True)
        self.background_only = background_only

    def clean_merged_data(self):
        """Function to clear cache of all data"""
        # remove all files inside merged_path using Pathlib
        if self.merged_path.exists():
            for file in self.merged_path.iterdir():
                if file.is_file():
                    file.unlink()
            logger.debug(f"Removed all files from {self.merged_path}")

    def get_subdirs(self):
        return [
            x
            for x in os.listdir(self.path)
            if x[0] != "." and x != "merged"
        ]

    def merge_datadict(self, merged: dict[str, list | dict], pending_data: dict[str, list | dict]):
        """Merge the content of pending_data into merged."""
        for key, info in pending_data.items():
            if isinstance(info, list):
                # Append the list to the existing one.
                merged[key] += info
            elif isinstance(info, dict):
                for param_name, params in info.items():
                    try:
                        merged[key][param_name] += params
                    except KeyError as m:
                        logger.warning(
                            f"Keys [{key}][{param_name}] not found in \n {merged}"
                        )
                        raise KeyError(m)
            else:
                raise TypeError(
                    f"Unexpected type for key {key}: {type(info)}. Expected list or dict."
                )


    def merge_and_load_subdir(self, subdir_name):
        """Merge and load data from a single subdirectory."""
        subdir = os.path.join(self.path, subdir_name)

        files = os.listdir(subdir)

        # Map one dir to one pickle
        merged_file_path = os.path.join(self.merged_path, subdir_name + ".pkl")
        # Load previously merged data, if it exists.
        if os.path.isfile(merged_file_path):
            logger.debug(f"loading merged data from {merged_file_path}")
            with open(merged_file_path, "rb") as mp:
                merged_data = Pickle.load(mp)
        else:
            merged_data = {}

        for filename in files:
            pending_file = os.path.join(subdir, filename)

            try:
                with open(pending_file, "rb") as f:
                    data = Pickle.load(f)
            except (EOFError, IsADirectoryError):
                logger.warning("Failed loading: {0}".format(pending_file))
                continue
            # Remove file immediately. This can lead to undesired results because if the program crashes or gets terminated in the process, we will have removed files before writing the merged data. However, delaying the removal would create the opposite problem: the file is merged but not removed, and potentially merged a second time at the next run.
            os.remove(pending_file)

            if merged_data == {}:
                merged_data = data
            else:
                self.merge_datadict(merged_data, data)

        # Save merged data.
        with open(merged_file_path, "wb") as mp:
            Pickle.dump(merged_data, mp)

        return merged_data

    def merge_and_load(self, output_dict: dict):
        # Loop over all injection scale subdirectories.
        scales_subdirs = self.get_subdirs()

        background_label = scale_shortener(0.0)

        for subdir_name in scales_subdirs:
            scale_label = scale_shortener(float(subdir_name))

            if self.background_only and scale_label != background_label:
                # skip non-background trials
                continue

            pending_data = self.merge_and_load_subdir(subdir_name)

            if pending_data:
                if scale_label == background_label and background_label in output_dict:
                    self.merge_datadict(output_dict[background_label], pending_data)
                else:
                    output_dict[scale_label] = pending_data



class ResultsHandler(object):
    def __init__(
        self,
        rh_dict,
        do_sens=True,
        do_disc=True,
        bias_error="std",
        sigma_thresholds=[3.0, 5.0],
        background_from=None
    ):
        self.sources = load_catalogue(rh_dict["catalogue"])

        self.name = rh_dict["name"]

        if background_from is not None:
            self.background_from = background_from
        else:
            self.background_from = rh_dict["name"]

        self.mh_name = rh_dict["mh_name"]

        self._inj_dict = rh_dict["inj_dict"]
        self._dataset = rh_dict["dataset"]

        # set the maximum number of function evaluations for the minimizer when doing the sensitivity / DP fit
        # 800 is the default scipy value but in some cases it might be necessary to increase it
        self.maxfev = rh_dict.get("maxfev", 800)

        self.results = dict()

        self.pickle_output_dir = name_pickle_output_dir(self.name)
        self.pickle_output_dir_bg = name_pickle_output_dir(self.background_from)

        self.pickle_cache = PickleCache(Path(self.pickle_output_dir))
        self.pickle_cache_bg = PickleCache(Path(self.pickle_output_dir_bg))

        self.plot_path = Path(plot_output_dir(self.name))

        self.allow_extrapolation = rh_dict.get("allow_extrapolated_sensitivity", True)

        self.valid = True

        self.bias_error = bias_error

        # ts_type reads 'flare' or 'standard' depending on 'mh_name'
        self.ts_type: str = get_ts_fit_type(rh_dict)

        # alternative "flare_plots" routine has been unmaintained and commented out for some time
        self.make_plots: callable = self.standard_plots

        # (p0, bounds, names)
        self.param_info: tuple = MinimisationHandler.find_parameter_info(rh_dict)

        # this will have the TS threshold values as keys and a tuple containing
        # (injection scale, relative overfluctuations, error on overfluctuations)
        # as values
        self.overfluctuations = dict()

        # Generalised discovery potential
        self.discovery = {}
        for threshold in sigma_thresholds:
            self.discovery[threshold] = {
                "ts": np.nan,
                "flux_val": np.nan,
                "flux_err": np.nan,
                "extrapolated": False,
            }
        self.discovery["nominal"] = {
            "ts": 25.0,
            "flux_val": np.nan,
            "flux_err": np.nan,
            "extrapolated": False,
        }

        # Load injection ladder values.
        # Builds a dictionary mapping the injection scale to the content of the trials.
        try:
            self.inj = self.load_injection_values()
        except FileNotFoundError as err:
            logger.error(
                "Unable to load injection values. Have you run this analysis at least once?"
            )
            logger.error(err)
            self.valid = False
            return

        try:
            self.pickle_cache.merge_and_load(output_dict=self.results)
            # Load the background trials. Will override the existing one.
            self.pickle_cache_bg.merge_and_load(output_dict=self.results)
        except FileNotFoundError:
            logger.warning(f"No files found at {self.pickle_output_dir}")

        # auxiliary parameters
        self.scale_values = sorted(
            [float(j) for j in self.results.keys()]
        )  
        self.scale_labels = [scale_shortener(i) for i in self.scale_values]

        logger.info(f"Injection scales: {self.scale_values}")
        # print("Scale labels: ", self.scale_labels)

        # Determine the injection scales
        try:
            self.find_ns_scale()
            self.plot_bias()
        except ValueError as e:
            logger.warning(f"RuntimeError for ns scale factor: \n {e}")
        except IndexError as e:
            logger.warning(
                f"IndexError for ns scale factor. Only background trials? \n {e}"
            )

        if do_sens:
            try:
                self.find_sensitivity()
            except ValueError as e:
                logger.warning(f"RuntimeError for sensitivity: \n {e}")

        if do_disc:
            try:
                self.find_disc_potential()
            except RuntimeError as e:
                logger.warning(f"RuntimeError for discovery potential: \n {e}")
            except TypeError as e:
                logger.warning(f"TypeError for discovery potential: \n {e}")
            except ValueError as e:
                logger.warning(f"ValueError for discovery potential: \n {e}")

        # attributes for backward compatibility
        self.disc_potential = self.discovery[5.0]["flux_val"]
        self.disc_err = self.discovery[5.0]["flux_err"]
        self.disc_potential_25 = self.discovery["nominal"]["flux_val"]
        self.disc_ts_threshold = self.discovery[5.0]["ts"]
        self.extrapolated_sens = self.discovery[5.0]["extrapolated"]
        self.extrapolated_disc = self.discovery[5.0]["extrapolated"]

    def is_valid(self):
        """If results are valid, returns True.
            If something went wrong during the instantiation, returns False.

        Returns:
            bool: whether results are valid or not.
        """
        return self.valid

    def __str__(self):
        out = f"Analysis result for `{self.name}`\n"
        if self.valid:
            extrapolated = (
                "extrapolated" if self.extrapolated_sens else "not extrapolated"
            )
            out += f"Sensitivity = {self.sensitivity:.2e} (+{self.sensitivity_err[0]:.2e}/-{self.sensitivity_err[1]:.2e}) [{extrapolated}]\n"
            out += f"Discovery potential (5 sigma from TS distribution) = {self.disc_potential:.2e}\n"
            out += f"Discovery potential (TS = 25) = {self.disc_potential_25:.2e}"
        else:
            out += "Result is invalid. Check the log messages."
        return out

    @property
    def ns(self):
        """returns the injection scales converted to number of signal neutrinos"""
        ns = (
            np.array([k_to_flux(float(s)) for s in self.scale_labels]) * self.flux_to_ns
        )
        return ns

    @property
    def ts_arrays(self):
        """returns the generated test statistic distributions as arrays for each injection step"""
        return [np.array(self.results[scale]["TS"]) for scale in self.scale_labels]

    @property
    def ns_injected(self):
        """returns the median of the injected number of signal neutrinos for each injection step"""
        ns_arrays = np.array(
            [
                np.array(
                    [
                        np.median(self.results[scale]["Parameters"][key])
                        for key in self.results[scale]["Parameters"]
                        if "n_s" in key
                    ]
                )
                for scale in self.scale_labels
            ]
        )

        # In the case of fitted weights there will be a number of injected neutrinos for each source thus we have
        # to take the sum. If this is not the case this won't do anything as ns_array will only have one entry.
        return [sum(a) for a in ns_arrays]

    def mean_injection_time(self):
        inj_time_list = list()
        for src in self.sources:
            single_inj_time = 0
            for s_name in self._dataset.keys():
                s = self._dataset[s_name]
                tpdf = TimePDF.create(
                    self._inj_dict["injection_sig_time_pdf"], s.get_time_pdf()
                )
                single_inj_time += tpdf.raw_injection_time(src)
            inj_time_list.append(single_inj_time)
        return np.median(inj_time_list)

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
        return calculate_astronomy(flux, e_pdf_dict)

    def clean_merged_data(self):
        """Clean merged data from pickle cache, only for main analysis. Do not touch the background cache."""
        self.pickle_cache.clean_merged_data()


    def load_injection_values(self):
        """Function to load the values used in injection, so that a
        comparison to the fit results can be made.

        :return: Dictionary of injected values.
        """

        load_dir = inj_dir_name(self.name)

        inj_values = dict()

        for file in os.listdir(load_dir):
            path = os.path.join(load_dir, file)

            if os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        inj_values[os.path.splitext(file)[0]] = Pickle.load(f)
                except EOFError as e:
                    logger.warning(f"{path}: EOFError: {e}! Can not use this scale!")

            else:
                logger.debug(f"Did not load {path}, not a file!")

        return inj_values


    def merge_and_load_pickle_data(self):
        # NOTE:
        # self.pickle_output_path
        # self.merged_dir = self.pickle_output_path / "merged"


        # Loop over all subdirectories, one for each injection scale, containing one pickle per trial.
        all_sub_dirs = [
            x
            for x in os.listdir(self.path)
            if x[0] != "." and x != "merged"
        ]
        # Create a "merged" directory, that will contain a single pickle with many trials per injection scale.
        try:
            os.makedirs(self.merged_dir)
        except OSError:
            pass

        for sub_dir_name in all_sub_dirs:
            sub_dir = os.path.join(self.path, sub_dir_name)

            files = os.listdir(sub_dir)

            # Map one dir to one pickle
            merged_path = os.path.join(self.merged_dir, sub_dir_name + ".pkl")
            # Load previously merged data, if it exists.
            if os.path.isfile(merged_path):
                logger.debug(f"loading merged data from {merged_path}")
                with open(merged_path, "rb") as mp:
                    merged_data = Pickle.load(mp)
            else:
                merged_data = {}

            for filename in files:
                pending_file = os.path.join(sub_dir, filename)

                try:
                    with open(pending_file, "rb") as f:
                        data = Pickle.load(f)
                except (EOFError, IsADirectoryError):
                    logger.warning("Failed loading: {0}".format(pending_file))
                    continue
                # This can be "dangerous" because if the program crashes or gets terminated, we will have removed files before writing the merged data.
                os.remove(pending_file)

                if merged_data == {}:
                    merged_data = data
                else:
                    for key, info in data.items():
                        if isinstance(info, list):
                            merged_data[key] += info
                        else:
                            for param_name, params in info.items():
                                try:
                                    merged_data[key][param_name] += params
                                except KeyError as m:
                                    logger.warning(
                                        f"Keys [{key}][{param_name}] not found in \n {merged_data}"
                                    )
                                    raise KeyError(m)

            # Save merged data.
            with open(merged_path, "wb") as mp:
                Pickle.dump(merged_data, mp)

            # Load merged data in results.
            if merged_data:
                self.results[scale_shortener(float(sub_dir_name))] = merged_data

        if not self.results:
            logger.warning("No data was found by ResultsHandler object! \n")
            logger.warning(
                "Tried root directory: \n {0} \n ".format(self.pickle_output_dir)
            )
            sys.exit()

    def find_ns_scale(self):
        """Find the number of neutrinos corresponding to flux"""
        try:
            # if weights were not fitted, number of neutrinos is stored in just one parameter

            if "n_s" in self.inj[self.scale_labels[1]]:
                self.flux_to_ns = self.inj[self.scale_labels[1]]["n_s"] / k_to_flux(
                    self.scale_values[1]
                )

            # if weights were fitted, or for cluster search, there is one n_s for each fitted source
            else:
                sc_dict = self.inj[self.scale_labels[1]]
                self.flux_to_ns = sum(
                    [sc_dict[k] for k in sc_dict if "n_s" in str(k)]
                ) / k_to_flux(self.scale_values[1])

            logger.debug(f"Conversion ratio of flux to n_s: {self.flux_to_ns:.2f}")

        except KeyError as e:
            logger.warning(
                f'KeyError: key "n_s" not found and minimizer is {self.mh_name}!!: {e}'
            )

    def estimate_sens_disc_scale(self):
        results = []

        logger.debug("  scale   avg_sigma     avg_TS")
        logger.debug("  ----------------------------")

        for scale, ts_array in zip(self.scale_values, self.ts_arrays):
            # calculate averages
            avg_ts = ts_array.sum() / ts_array.size
            avg_sigma = np.sqrt(avg_ts)

            # error on average sigma is 1 sigma/sqrt(trials.size)
            err_sigma = 1.0 / np.sqrt(ts_array.size)

            # collect all sigma > 0
            if avg_sigma >= 0:
                logger.debug(
                    f"  {scale:.4f}   {avg_sigma:.2f}+/-{err_sigma:.2f}  {avg_ts:.4f}"
                )
                results.append([scale, avg_sigma, err_sigma, avg_ts])
            else:
                pass

        results = np.transpose(results)

        # linear fit
        p = np.polyfit(
            results[0],  # x = scale
            results[1],  # y = avg. sigma
            1,  # 1st order poly
            w=results[2],
        )  # error = error on avg. sigma

        # discovery threshold is 5 sigma
        disc_scale_guess = (5 - p[1]) / p[0]

        # sensitivity threshold is usually ~0.3 x discovery
        sens_scale_guess = 0.3 * disc_scale_guess

        # make a plot
        fig, ax = plt.subplots()
        # plot injection results
        ax.errorbar(
            results[0],
            results[1],
            yerr=results[2],
            ls="",
            color="k",
            label="quick injections",
        )
        # plot the linear fit
        xplot = np.linspace(min(results[0]), max(results[0]), 100)
        yplot = xplot * p[0] + p[1]
        ax.plot(xplot, yplot, marker="", label="linear fit")
        # plot guessed scales
        ax.axvline(disc_scale_guess, ls="--", color="red", label="DP scale guess")
        ax.axvline(sens_scale_guess, ls="--", color="blue", label="Sens scale guess")
        ax.set_xlabel("flux scale")
        ax.set_ylabel(r"$\sigma_{mean}$")
        ax.legend()
        fn = self.plot_path / "quick_injection_scale_guess.pdf"
        fig.savefig(fn)
        logger.debug(f"saved figure under {fn}")
        plt.close()

        logger.debug(
            f"disc scale guess: {disc_scale_guess}; sens scale guess: {sens_scale_guess}"
        )

        return disc_scale_guess, sens_scale_guess

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

        savepath = self.plot_path / "sensitivity.pdf"

        (
            self.sensitivity,
            self.sensitivity_err,
            self.extrapolated_sens,
        ) = self.find_overfluctuations(bkg_median, savepath)

        msg = "EXTRAPOLATED " if self.extrapolated_sens else ""
        logger.info(f"{msg}Sensitivity is {self.sensitivity:.3g}")

    def find_overfluctuations(self, ts_val, savepath=None):
        """Uses the values of injection trials to fit an 1-exponential decay
        function to the fraction of overfluctuations above `ts_val` as a function
        of the injected flux (or n_s).

        For ts_val equal to the background median, this allows to calculate the sensitivity.
        For ts_val equal to an unblinded value of the TS, this allows to calculate an upper limit.

        Where the injected flux was not sufficient to reach the
        sensitivity, extrapolation is be used instead of interpolation,
        but this will obviously have larger associated errors. If
        extrapolation is used, the third return value is set to True. In
        either case, a plot of the overfluctuations as a function of the
        injected signal is produced.
        """
        x_acc, y, yerr = [], [], []
        x = self.scale_labels

        for scale in x:
            ts_array = np.array(self.results[scale]["TS"])
            frac = float(len(ts_array[ts_array > ts_val])) / (float(len(ts_array)))

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
                yerr.append(1.0 / np.sqrt(float(len(ts_array))))

                if frac != 0.0:
                    logger.info(f"Making plot for {scale=}, {frac=}")
                    self.make_plots(scale)
                else:
                    logger.warning(
                        f"Fraction of overfluctuations is {frac=}, skipping plot for {scale=}"
                    )
        if len(np.where(np.array(y) < 0.95)[0]) < 2:
            raise OverfluctuationError(
                "Not enough points with overfluctuations under 95%, lower injection scale!"
            )

        x = np.array(x_acc)
        self.overfluctuations[ts_val] = x, y, yerr

        fit, err, extrapolated = self.sensitivity_fit(savepath, ts_val)
        return fit, err, extrapolated

    def sensitivity_fit(self, savepath, ts_val):
        x, y, yerr = self.overfluctuations[ts_val]
        x_flux = k_to_flux(x)

        threshold = 0.9

        b = 1 - min(y)

        def f(x, a):
            value = 1 - b * np.exp(-a * x)
            return value

        popt, pcov = scipy.optimize.curve_fit(
            f,
            x,
            y,
            sigma=yerr,
            absolute_sigma=True,
            p0=[1.0 / max(x)],
            maxfev=self.maxfev,
        )

        perr = np.sqrt(np.diag(pcov))

        best_a = popt[0]

        def best_f(x, sd=0.0):
            a = best_a + perr * sd
            return f(x, a)

        fit = k_to_flux((1.0 / best_a) * np.log(b / (1 - threshold)))

        if fit > max(x_flux):
            extrapolation_msg = (
                "The sensitivity is beyond the range of the tested scales."
                "The number is probably not good."
            )
            if self.allow_extrapolation:
                logger.warning(extrapolation_msg)
                extrapolated = True
            else:
                raise OverfluctuationError(extrapolation_msg)
        else:
            extrapolated = False

        xrange = np.linspace(0.0, 1.1 * max(x), 1000)

        lower = k_to_flux((1.0 / (best_a + perr)) * np.log(b / (1 - threshold)))
        upper = k_to_flux((1.0 / (best_a - perr)) * np.log(b / (1 - threshold)))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.errorbar(x_flux, y, yerr=yerr, color="black", fmt=" ", marker="o")
        ax1.plot(k_to_flux(xrange), best_f(xrange), color="blue")
        ax1.fill_between(
            k_to_flux(xrange),
            best_f(xrange, 1),
            best_f(xrange, -1),
            color="blue",
            alpha=0.1,
        )
        ax1.axhline(threshold, lw=1, color="red", linestyle="--")
        ax1.axvline(fit, lw=2, color="red")
        ax1.axvline(lower, lw=2, color="red", linestyle=":")
        ax1.axvline(upper, lw=2, color="red", linestyle=":")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlim(0.0, k_to_flux(max(xrange)))
        ax1.set_ylabel("Overfluctuations above TS=" + "{:.2f}".format(ts_val))
        plt.xlabel(r"Flux Normalisation @ 1GeV [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")

        if not np.isnan(self.flux_to_ns):
            ax2 = ax1.twiny()
            ax2.grid(0)
            ax2.set_xlim(0.0, self.flux_to_ns * k_to_flux(max(xrange)))
            ax2.set_xlabel(r"Number of neutrinos")

        fig.savefig(savepath)
        plt.close()

        if len(np.where(np.array(y) < 0.95)[0]) < 2:
            raise OverfluctuationError(
                "Not enough points with overfluctuations under 95%, lower injection scale!"
            )

        fit_err = np.array([fit - lower, upper - fit]).T[0]

        return fit, fit_err, extrapolated

    def find_disc_potential(self):
        ts_path = self.plot_path / "ts_distributions/0.pdf"

        try:
            bkg_dict = self.results[scale_shortener(0.0)]
        except KeyError:
            logger.error("No key equal to '0'")
            return

        bkg_ts = bkg_dict["TS"]

        # determine the ts values corresponding to the different significance (z) values
        for zval in self.discovery:
            # skip the "nominal" TS=25 case
            if zval != "nominal":
                self.discovery[zval]["ts"] = plot_background_ts_distribution(
                    bkg_ts, ts_path, ts_type=self.ts_type, zval=zval
                )

        x = self.scale_labels

        # maps sigma values to array of overfluctuation fractions for the injection strength represented in x
        y = {zval: [] for zval in self.discovery}

        # evaluate for each injection step the fraction of overfluctuations above the threshold
        # for each of the z values
        for scale in x:
            ts_array = np.array(self.results[scale]["TS"])

            for zval in self.discovery:
                disc_threshold = self.discovery[zval]["ts"]

                if not np.isnan(disc_threshold):
                    frac = float(len(ts_array[ts_array > disc_threshold])) / (
                        float(len(ts_array))
                    )

                    logger.info(
                        f"Scale: {scale}, TS_threshold: {disc_threshold}, n_trials: {len(ts_array)} => overfluctuations {frac=}"
                    )

                    y[zval].append(frac)
                else:
                    logger.warning(
                        f"Invalid (NaN) discovery threshold for {zval}-sigma, this will be ingnored."
                    )

            self.make_plots(scale)

        x = np.array([float(s) for s in x])
        x_flux = k_to_flux(x)

        threshold = 0.5

        """
        Now calculate discovery flux.
        """
        discovery_flux = {}

        for zval in self.discovery:
            # if the threshold is nan, skip
            if np.isnan(self.discovery[zval]["ts"]):
                discovery_flux[zval] = np.nan

            y_vals = y[zval]

            def f(x, a, b, c):
                value = scipy.stats.gamma.cdf(x, a, b, c)
                return value

            # this trick could be replaced by calling f on the vector of best fit parameters
            best_f = None

            try:
                res = scipy.optimize.curve_fit(
                    f,
                    x,
                    y_vals,
                    p0=[6, -0.1 * max(x), 0.1 * max(x)],
                    maxfev=self.maxfev,
                )

                best_a = res[0][0]
                best_b = res[0][1]
                best_c = res[0][2]

                def best_f(x):
                    return f(x, best_a, best_b, best_c)

                # estimate the solution flux
                interpolated_flux = scipy.stats.gamma.ppf(0.5, best_a, best_b, best_c)

                # "disc_potential" and "disc_potential_25" attributes are set here
                # use of `setattr` makes the code a bit obscure and could be improved
                discovery_flux[zval] = interpolated_flux

            except RuntimeError as e:
                logger.warning(f"RuntimeError for discovery potential!: {e}")
                # interpolated_flux = np.nan

            # now plot the whole ordeal
            xrange = np.linspace(0.0, 1.1 * max(x), 1000)

            save_path = self.plot_path / f"disc_{zval}.pdf"

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(x_flux, y_vals, color="black")

            if not best_f is not None:
                ax1.plot(k_to_flux(xrange), best_f(xrange), color="blue")

            ax1.axhline(threshold, lw=1, color="red", linestyle="--")
            ax1.axvline(self.sensitivity, lw=2, color="black", linestyle="--")
            if not np.isnan(interpolated_flux):
                ax1.axvline(interpolated_flux, lw=2, color="red")
            ax1.set_ylim(0.0, 1.0)
            ax1.set_xlim(0.0, k_to_flux(max(xrange)))
            ax1.set_ylabel(r"Overfluctuations relative to f{zval}$\sigma$ threshold")
            plt.xlabel(r"Flux Normalisation @ 1GeV [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")

            if not np.isnan(self.flux_to_ns):
                ax2 = ax1.twiny()
                ax2.grid()
                ax2.set_xlim(0.0, self.flux_to_ns * k_to_flux(max(xrange)))
                ax2.set_xlabel(r"Number of neutrinos")

            fig.savefig(save_path)
            plt.close()

            extrapolated = interpolated_flux > max(x_flux)

            logger.info(
                f"Discovery Potential ({zval}-sigma): {discovery_flux[zval]} ({extrapolated=})"
            )

            self.discovery[zval]["flux_val"] = discovery_flux[zval]
            self.discovery[zval]["extrapolated"] = extrapolated

    def standard_plots(self, scale):
        ts_array = np.array(self.results[scale]["TS"])
        ts_path = self.plot_path / f"ts_distributions/{str(scale)}.pdf"

        plot_background_ts_distribution(ts_array, ts_path, ts_type=self.ts_type)

        param_path = self.plot_path / f"params/{str(scale)}.pdf"

        # if self.show_inj:
        try:
            inj = self.inj[str(scale)]

            plot_fit_results(self.results[scale]["Parameters"], param_path, inj=inj)
        except KeyError as e:
            logger.warning(
                f"KeyError for scale {scale}: {e}! Can not plot fit results!"
            )

    def ts_evolution_gif(self, n_scale_steps=None, cmap_name="winter"):
        logger.debug("making animation")

        all_scales_list = list(self.results.keys())
        n_scales_all = len(all_scales_list)

        n_scale_steps = n_scales_all - 1 if not n_scale_steps else n_scale_steps

        scale_step_length = int(round(n_scales_all / (n_scale_steps)))
        scales = [
            all_scales_list[min([i * scale_step_length, n_scales_all - 1])]
            for i in range(n_scale_steps + 1)
        ]

        ts_arrays = [np.array(self.results[scale]["TS"]) for scale in scales]

        ns_arrays = np.array(
            [
                np.array(
                    [
                        np.median(self.results[scale]["Parameters"][key])
                        for key in self.results[scale]["Parameters"]
                        if "n_s" in key
                    ]
                )
                for scale in scales
            ]
        )

        n_s = [sum(a) for a in ns_arrays]
        logger.debug("numbers of injected neutrinos: " + str(n_s))

        norm = colors.Normalize(vmin=0, vmax=max(n_s))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        cmap = mappable.get_cmap()

        sq_fig, sq_ax = plt.subplots()
        sq_fig.set_tight_layout(True)
        sq_ax.set_xlim([-5, max(ts_arrays[-1]) + 10])
        sq_ax.set_yscale("log")
        sq_ax.set_xlabel("Test Statistic")
        sq_ax.set_ylabel("a.u.")

        sqbar = sq_fig.colorbar(mappable, ax=sq_ax)
        sqbar.set_label(r"n$_{\mathrm{injected}}$")

        def update(i):
            its = ts_arrays[i]
            ins = n_s[i]
            sq_ax.hist(
                its, histtype="stepfilled", density=True, color=cmap(ins / max(n_s))
            )
            sq_ax.set_title(r"n$_{\mathrm{injected}}=$" + "{:.2f}".format(ins))

        anim = animation.FuncAnimation(
            sq_fig, update, frames=np.arange(0, n_scale_steps), interval=500
        )

        anim_name = self.plot_path / "ts_distributions/ts_distributions_evolution.gif"
        logger.debug(f"saving animation under {anim_name}")
        anim.save(anim_name, dpi=80, writer="imagemagick")

    def ts_distribution_evolution(self):
        logger.debug("plotting evolution of TS distribution")

        all_scales = np.array(list(self.results.keys()))
        all_scales_floats = [float(sc) for sc in all_scales]

        logger.debug("all scales: " + str(all_scales_floats))
        logger.debug("sensitivity scale: " + str(flux_to_k(self.sensitivity)))

        sens_scale = all_scales[
            all_scales_floats >= np.array(flux_to_k(self.sensitivity))
        ][0]
        disc_scale = all_scales[
            all_scales_floats >= np.array(flux_to_k(self.disc_potential))
        ][0]

        scales = [all_scales[0], sens_scale, disc_scale]
        ts_arrays = [np.array(self.results[scale]["TS"]) for scale in scales]
        ns_arrays = np.array(
            [
                np.array(
                    [
                        np.median(self.results[scale]["Parameters"][key])
                        for key in self.results[scale]["Parameters"]
                        if "n_s" in key
                    ]
                )
                for scale in scales
            ]
        )

        n_s = [sum(a) for a in ns_arrays]
        logger.debug("numbers of injected neutrinos: " + str(n_s))

        fig, ax = plt.subplots()

        ax.hist(
            ts_arrays[0],
            histtype="stepfilled",
            label="background",
            density=True,
            alpha=0.6,
            color="blue",
        )

        ax.hist(
            ts_arrays[1],
            histtype="step",
            density=True,
            color="orange",
            label="signal: {:.2} signal neutrinos".format(n_s[1]),
        )
        ax.axvline(
            self.bkg_median, ls="--", label="sensitivity threshold", color="orange"
        )

        ax.hist(
            ts_arrays[2],
            histtype="step",
            density=True,
            color="red",
            label="signal: {:.2} signal neutrinos".format(n_s[2]),
        )
        ax.axvline(
            self.disc_ts_threshold,
            ls="--",
            label="discovery potential threshold",
            color="red",
        )

        ax.set_xlabel("Test Statistic")
        ax.set_ylabel("a.u.")
        ax.legend()
        ax.set_yscale("log")

        plt.tight_layout()

        sn = self.plot_path / "ts_distributions/ts_evolution_.pdf"
        logger.debug(f"saving plot to {sn}")
        fig.savefig(sn)

        plt.close()

    def plot_bias(self):
        raw_x = self.scale_labels
        base_x = [k_to_flux(float(j)) for j in raw_x]
        base_x_label = r"$\Phi_{1GeV}$ (GeV$^{-1}$ cm$^{-2}$)"

        (_p0, _bounds, names) = self.param_info

        for i, param in enumerate(names):
            try:
                plt.figure()

                ax = plt.subplot(111)

                meds = []
                ulims = []
                llims = []
                trues = []

                for scale in raw_x:
                    vals = self.results[scale]["Parameters"][param]

                    if self.bias_error == "std":
                        med = np.median(vals)
                        meds.append(med)
                        sig = np.std(vals)
                        ulims.append(med + sig)
                        llims.append(med - sig)

                    elif self.bias_error == "ci90":
                        med, llim, ulim = np.quantile(vals, [0.5, 0.05, 0.95])
                        meds.append(med)
                        llims.append(llim)
                        ulims.append(ulim)

                    else:
                        raise ValueError(
                            f"Invalid value {self.bias_error} for bias_error!"
                        )

                    true = self.inj[scale][param]
                    trues.append(true)

                do_ns_scale = False

                if "n_s" in param:
                    x = trues
                    x_label = r"$n_{injected}$" + param.replace("n_s", "")
                else:
                    x = base_x
                    x_label = base_x_label

                # decide wether to plot a second x axis on the top axis indicating the number of injected
                # neutrinos instead of the flux
                if "gamma" in param:
                    if not isinstance(self.flux_to_ns, type(None)):
                        do_ns_scale = True

                ns_scale = ns_scale_label = None

                if do_ns_scale:
                    ns_scale = self.flux_to_ns * max(base_x)
                    ns_scale_label = "Number of neutrinos"

                plt.scatter(x, meds, color="orange")
                plt.plot(x, meds, color="black")
                plt.plot(x, trues, linestyle="--", color="red")
                plt.fill_between(x, ulims, llims, alpha=0.5, color="orange")

                try:
                    ax.set_xlim(left=0.0, right=max(x))
                    if min(trues) == 0.0:
                        ax.set_ylim(bottom=0.0)

                    if do_ns_scale:
                        ax2 = ax.twiny()
                        ax2.grid(0)
                        ax2.set_xlim(0.0, ns_scale)
                        ax2.set_xlabel(ns_scale_label)
                except ValueError as e:
                    logger.warning(f"{param}: {e}")

                ax.set_xlabel(x_label)
                ax.set_ylabel(param)
                plt.title("Bias (" + param + ")")

                savepath = self.plot_path / f"bias_{param}.pdf"
                logger.info("Saving bias plot to {0}".format(savepath))

                try:
                    os.makedirs(os.path.dirname(savepath))
                except OSError:
                    pass

                plt.tight_layout()
                plt.savefig(savepath)

            except KeyError as e:
                logger.warning(f"KeyError for {param}: {e}! Can not make bias plots!")

            finally:
                plt.close()

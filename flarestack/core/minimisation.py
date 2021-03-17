import logging
import numpy as np
import resource
import random
from sys import stdout
import os
import argparse
import pickle as Pickle
import scipy.optimize
from flarestack.core.injector import read_injector_dict
from flarestack.core.llh import LLH, generate_dynamic_flare_class, read_llh_dict
from flarestack.shared import name_pickle_output_dir, \
    inj_dir_name, plot_output_dir, scale_shortener, flux_to_k
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
import matplotlib as mpl
from flarestack.core.time_pdf import TimePDF, Box, Steady
from flarestack.core.angular_error_modifier import BaseAngularErrorModifier
from flarestack.utils.catalogue_loader import load_catalogue, \
    calculate_source_weight
from flarestack.utils.asimov_estimator import estimate_discovery_potential

logger = logging.getLogger(__name__)

def time_smear(inj):
    inj_time = inj["injection_sig_time_pdf"]
    max_length = inj_time["max_offset"] - inj_time["min_offset"]
    offset = np.random.random() * max_length + inj_time["min_offset"]
    inj_time["offset"] = offset
    return inj_time


def read_mh_dict(mh_dict):
    """Ensure backwards compatibility of MinimisationHandler dictionary objects

    :param mh_dict: MinimisationHandler dictionary
    :return: MinimisationHandler dictionary compatible with new format
    """

    # Ensure backwards compatibility

    maps = [
        ("inj kwargs", "inj_dict"),
        ("datasets", "dataset"),
        ("background TS", "background_ts")
    ]

    for (old_key, new_key) in maps:

        if old_key in list(mh_dict.keys()):
            logger.warning("Deprecated mh_dict key '{0}' was used. Please use '{1}' in future.".format(
                old_key, new_key))
            mh_dict[new_key] = mh_dict[old_key]

    if "name" not in mh_dict.keys():
        raise KeyError("mh_dict object is missing key 'name'."
                       "This should be the unique save path for results.")

    elif mh_dict["name"][-1] != "/":
        mh_dict["name"] += "/"

    pairs = [
        ("inj_dict", read_injector_dict),
        ("llh_dict", read_llh_dict)
    ]

    for (key, f) in pairs:
        if key in list(mh_dict.keys()):
            mh_dict[key] = f(mh_dict[key])

    if np.logical_and("fixed_scale" in mh_dict.keys(), "n_steps" in mh_dict.keys()):
        raise Exception(f"MinimisationHandler dictionary contained both 'fixed_scale' key for "
                        f"set injection flux, and 'n_steps' key for stepped injection flux."
                        f"Please use only one of these options. \n  mh_dict: \n {mh_dict}")

    return mh_dict


class MinimisationHandler(object):
    """Generic Class to handle both dataset creation and llh minimisation from
    experimental data and Monte Carlo simulation. Initialised with a set of
    IceCube datasets, a list of sources, and independent sets of arguments for
    the injector and the likelihood.
    """
    subclasses = {}

    # Each MinimisationHandler must specify which LLH classes are compatible
    compatible_llh = []
    compatible_negative_n_s = False

    def __init__(self, mh_dict):

        mh_dict = read_mh_dict(mh_dict)

        sources = load_catalogue(mh_dict["catalogue"])

        self.name = mh_dict["name"]

        self.pickle_output_dir = name_pickle_output_dir(self.name)
        self._injectors = dict()
        self._llhs = dict()
        self._aem = dict()
        self.seasons = mh_dict["dataset"]
        self.sources = sources
        self.mh_dict = mh_dict

        if "inj_dict" in mh_dict.keys():

            # Checks whether signal injection should be done with a sliding PDF
            # within a larger window, or remain fixed at the specified time

            inj = dict(mh_dict["inj_dict"])

            try:
                self.time_smear = inj["injection_sig_time_pdf"]["time_smear_bool"]
            except KeyError:
                self.time_smear = False

            if self.time_smear:
                inj["injection_sig_time_pdf"] = time_smear(inj)

            self.inj_dict = inj

        # An independent set of Season objects can be used for the injector
        # This enables, for example, different MC sets to be used for
        # injection, to test the impact of different systematics

        try:
            self.inj_seasons = mh_dict["inj_dict"]["injection_dataset"]
            logger.debug("Using independent injection dataset.")

            if self.inj_seasons.keys() != self.seasons.keys():
                raise Exception("Key mismatch between injection and llh "
                                "Season objects. Injection Seasons have "
                                "keys:\n {0} \n and LLH Seasons have keys: \n"
                                "{1}". format(self.inj_seasons.keys(),
                                              self.seasons.keys()))

        except KeyError:
            self.inj_seasons = self.seasons

        self.llh_dict = mh_dict["llh_dict"]

        # Check if the specified MinimisationHandler is compatible with the
        # chosen LLH class

        if self.llh_dict["llh_name"] not in self.compatible_llh:
            raise ValueError("Specified LLH ({}) is not compatible with "
                             "selected MinimisationHandler".format(
                              self.llh_dict["llh_name"]))
        else:
            logger.info("Using '{0}' LLH class".format(self.llh_dict["llh_name"]))

        # Checks if negative n_s is specified for use, and whether this is
        # compatible with the chosen MinimisationHandler

        try:
            self.negative_n_s = self.llh_dict["negative_ns_bool"]
        except KeyError:
            self.negative_n_s = False

        if self.negative_n_s and not self.compatible_negative_n_s:
            raise ValueError("MinimisationHandler has been instructed to \n"
                             "allow negative n_s, but this is not compatible \n"
                             "with the selected MinimisationHandler.")

        # Sets up whether what pull corrector should be used (default is
        # none), and whether an angular error floor should be applied (
        # default is a static floor.

        try:
            self.pull_name = self.llh_dict["pull_name"]
        except KeyError:
            self.pull_name = "no_pull"

        try:
            self.floor_name = self.llh_dict["floor_name"]
        except KeyError:
            self.floor_name = "static_floor"

        p0, bounds, names = self.return_parameter_info(mh_dict)

        self.p0 = p0
        self.bounds = bounds
        self.param_names = names

        self.disc_guess = np.nan

    @classmethod
    def register_subclass(cls, mh_name):
        """Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """
        def decorator(subclass):
            cls.subclasses[mh_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, mh_dict):
        mh_dict = read_mh_dict(mh_dict)

        mh_name = mh_dict["mh_name"]

        if mh_name not in cls.subclasses:
            raise ValueError('Bad MinimisationHandler name {}'.format(mh_name))

        return cls.subclasses[mh_name](mh_dict)

    @classmethod
    def find_parameter_info(cls, mh_dict):
        read_mh_dict(mh_dict)
        mh_name = mh_dict["mh_name"]

        if mh_name not in cls.subclasses:
            raise ValueError('Bad MinimisationHandler name {}'.format(mh_name))

        return cls.subclasses[mh_name].return_parameter_info(mh_dict)

    def run_trial(self, full_dataset):
        pass

    def run(self, n_trials, scale=1., seed=None):
        pass

    @staticmethod
    def trial_params(mh_dict):

        if "fixed_scale" in list(mh_dict.keys()):
            scale_range = [mh_dict["fixed_scale"]]

        # elif mh_dict.get("background_only", False):
        #     # Only do the background trials
        #     # In this case only n_trials background trials are performed, not 10x n_trials!
        #     scale_range = np.array([0])
        #
        # elif mh_dict.get("injection_only", False):
        #     # Only do trials with signal injection
        #     scale = mh_dict["scale"]
        #     steps = int(mh_dict["n_steps"])
        #     scale_range = np.array(list(np.linspace(0., scale, steps)[1:]))

        else:
            scale = mh_dict["scale"]
            steps = int(mh_dict["n_steps"])
            background_ntrials_factor = mh_dict.get('background_ntrials_factor', 10)
            scale_range = np.array(
                [0. for _ in range(background_ntrials_factor)] +
                list(np.linspace(0., scale, steps)[1:])
            )

        n_trials = int(mh_dict["n_trials"])

        return scale_range, n_trials

    def iterate_run(self, scale=1., n_steps=5, n_trials=50):

        scale_range = np.linspace(0., scale, n_steps)[1:]

        self.run(n_trials*10, scale=0.0)

        for scale in scale_range:
            self.run(n_trials, scale)

    @staticmethod
    def return_parameter_info(mh_dict):
        seeds = []
        bounds = []
        names = []
        return seeds, names, bounds

    @staticmethod
    def return_injected_parameters(mh_dict):
        return {}

    def add_likelihood(self, season):
        return LLH.create(season, self.sources, self.llh_dict)

    def get_likelihood(self, season_name):

        if season_name not in self._llhs.keys():
            self._llhs[season_name] = self.add_likelihood(self.seasons[season_name])

        return self._llhs[season_name]

    def add_injector(self, season, sources):
        return season.make_injector(sources, **self.inj_dict)

    def get_injector(self, season_name):

        if season_name not in self._injectors.keys():
            self._injectors[season_name] = self.add_injector(self.seasons[season_name], self.sources)

        return self._injectors[season_name]

    def add_angular_error_modifier(self, season):
        return BaseAngularErrorModifier.create(
                season, self.llh_dict["llh_energy_pdf"], self.floor_name,
                self.pull_name,
                gamma_precision=self.llh_dict.get('gamma_precision', 'flarestack')
        )

    def get_angular_error_modifier(self, season_name):

        if season_name not in self._aem.keys():
            self._aem[season_name] = self.add_angular_error_modifier(self.seasons[season_name])

        return self._aem[season_name]

    @staticmethod
    def set_random_seed(seed):
        np.random.seed(seed)

    def guess_scale(self):
        """Method to guess flux scale for sensitivity + discovery potential
        :return:
        """
        return 1.5 * flux_to_k(self.guess_discovery_potential())

    def guess_discovery_potential(self):
        self.disc_guess = estimate_discovery_potential(
            self.seasons, dict(self.inj_dict), self.sources, dict(self.llh_dict))
        return self.disc_guess


@MinimisationHandler.register_subclass('fixed_weights')
class FixedWeightMinimisationHandler(MinimisationHandler):
    """Class to perform generic minimisations using a 'fixed weights' matrix.
    Sources are assigned intrinsic weights based on their assumed luminosity
    and/or distance, which are fixed. In addition, time weighting is used
    assuming a fixed fluence per source. The detector acceptance continues to
    vary as a function of the parameters given in minimisation step.
    """

    compatible_llh = ["spatial", "fixed_energy", "standard",
                      "standard_overlapping", "standard_matrix"]
    compatible_negative_n_s = True

    def __init__(self, mh_dict):

        MinimisationHandler.__init__(self, mh_dict)

        self.fit_weights = False

        # Checks if minimiser should be seeded from a brute scan

        try:
            self.brute = self.llh_dict["brute_seed"]
        except KeyError:
            self.brute = False

        # self.clean_true_param_values()

    def clear(self):

        self._injectors.clear()
        self._llhs.clear()

        del self

    def dump_results(self, results, scale, seed):
        """Takes the results of a set of trials, and saves the dictionary as
        a pickle pkl_file. The flux scale is used as a parent directory, and the
        pickle pkl_file itself is saved with a name equal to its random seed.

        :param results: Dictionary of Minimisation results from trials
        :param scale: Scale of inputted flux
        :param seed: Random seed used for running of trials
        """

        if self.name == " /":
            logger.warning("No field 'name' was specified in mh_dict object. "
                            "Cannot save results without a unique directory"
                            " name being specified.")

        else:

            write_dir = os.path.join(self.pickle_output_dir, scale_shortener(scale))

            # Tries to create the parent directory, unless it already exists
            try:
                os.makedirs(write_dir)
            except OSError:
                pass

            file_name = os.path.join(write_dir, str(seed) + ".pkl")

            logger.debug("Saving to {0}".format(file_name))

            with open(file_name, "wb") as f:
                Pickle.dump(results, f)

    def dump_injection_values(self, scale):

        if self.name == " /":
            raise Exception("No field 'name' was specified in mh_dict object. "
                            "Cannot save results without a unique directory"
                            " name being specified.")

        else:

            inj_dict = self.return_injected_parameters(scale)

            inj_dir = inj_dir_name(self.name)

            # Tries to create the parent directory, unless it already exists
            try:
                os.makedirs(inj_dir)
            except OSError:
                pass

            file_name = os.path.join(inj_dir, scale_shortener(scale) + ".pkl")

            logger.debug(f"Dumping Injection values to {file_name}")

            with open(file_name, "wb") as f:
                Pickle.dump(inj_dict, f)

    def run_trial(self, full_dataset):

        raw_f = self.trial_function(full_dataset)

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

        vals = res.x
        flag = res.status
        # If the minimiser does not converge, repeat with brute force
        if flag == 1:
            vals = scipy.optimize.brute(llh_f, ranges=self.bounds,
                                        finish=None)

        best_llh = raw_f(vals)

        if np.logical_and(not res.x[0] > 0.0, self.negative_n_s):

            bounds = list(self.bounds)
            bounds[0] = (-1000., -0.)
            start_seed = list(self.p0)
            start_seed[0] = -1.

            new_res = scipy.optimize.minimize(
                llh_f, start_seed, bounds=bounds)

            if new_res.status == 0:
                res = new_res

            vals = [res.x[0]]
            best_llh = res.fun

        ts = np.sum(best_llh)

        if ts == -0.0:
            ts = 0.0

        parameters = dict()

        for i, val in enumerate(vals):
            parameters[self.param_names[i]] = val

        res_dict = {
            "res": res,
            "Parameters": parameters,
            "TS": ts,
            "Flag": flag,
            "f": llh_f
        }

        return res_dict

    def run_single(self, full_dataset, scale, seed):

        param_vals = {}
        for key in self.param_names:
            param_vals[key] = []
        ts_vals = []
        flags = []

        res_dict = self.run_trial(full_dataset)

        for (key, val) in res_dict["Parameters"].items():
            param_vals[key].append(val)

        ts_vals.append(res_dict["TS"])
        flags.append(res_dict["Flag"])

        mem_use = str(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
        logger.debug('Memory usage max: {0} (Gb)'.format(mem_use))

        results = {
            "TS": ts_vals,
            "Parameters": param_vals,
            "Flags": flags,
        }

        self.dump_results(results, scale, seed)
        return res_dict

    def simulate_and_run(self, scale, seed=None):
        if seed is None:
            seed = np.random.randint(low=0, high=99999999)
        self.set_random_seed(seed)
        full_dataset = self.prepare_dataset(scale, seed)
        return self.run_single(full_dataset, scale, seed)

    def run(self, n_trials, scale=1., seed=None):

        if seed is None:
            seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        # param_vals = [[] for x in self.p0]
        param_vals = {}
        for key in self.param_names:
            param_vals[key] = []
        ts_vals = []
        flags = []

        logger.info("Generating {0} trials!".format(n_trials))

        for i in range(int(n_trials)):

            res_dict = self.simulate_and_run(scale)

            for (key, val) in res_dict["Parameters"].items():
                param_vals[key].append(val)

            ts_vals.append(res_dict["TS"])
            flags.append(res_dict["Flag"])

        n_inj = 0
        for season in self.seasons.keys():
            inj = self.get_injector(season)
            n_inj += np.sum(inj.n_exp["n_exp"] * scale)

        logger.info("Injected with an expectation of {0} events.".format(n_inj))

        logger.info("FIT RESULTS:")

        for (key, param) in sorted(param_vals.items()):
            if len(param) > 0:
                logger.info("Parameter {0}: {1} {2} {3}".format(key, np.mean(param),
                      np.median(param), np.std(param)))
        logger.info("Test Statistic: {0} {1} {2}".format(np.mean(ts_vals),
              np.median(ts_vals), np.std(ts_vals)))

        logger.info("FLAG STATISTICS:")
        for i in sorted(np.unique(flags)):
            logger.info("Flag {0}:{1}".format(i, flags.count(i)))

        results = {
            "TS": ts_vals,
            "Parameters": param_vals,
            "Flags": flags,
        }

        self.dump_results(results, scale, seed)
        self.dump_injection_values(scale)

    def make_season_weight(self, params, season):

        src = self.sources

        weight_scale = calculate_source_weight(src)

        # dist_weight = src["distance_mpc"] ** -2
        # base_weight = src["base_weight"]

        llh = self.get_likelihood(season.season_name)
        acc = []

        time_weights = []
        source_weights = []

        for source in src:
            time_weights.append(llh.sig_time_pdf.effective_injection_time(
                source))
            acc.append(llh.acceptance(source, params))
            source_weights.append(calculate_source_weight(source) /
                                  weight_scale)

        time_weights = np.array(time_weights)
        source_weights = np.array(source_weights)

        acc = np.array(acc).T[0]

        w = acc * time_weights
        w *= source_weights

        w = w[:, np.newaxis]

        return w

    def make_weight_matrix(self, params):

        # Creates a matrix fixing the fraction of the total signal that
        # is expected in each Source+Season pair. The matrix is
        # normalised to 1, so that for a given total n_s, the expectation
        # for the ith season for the jth source is given by:
        #  n_exp = n_s * weight_matrix[i][j]

        weights_matrix = np.ones([len(self.seasons), len(self.sources)])

        for i, season in enumerate(self.seasons.values()):
            w = self.make_season_weight(params, season)

            for j, ind_w in enumerate(w):
                weights_matrix[i][j] = ind_w

        return weights_matrix

    def prepare_dataset(self, scale=1., seed=None):

        if seed is None:
            seed = int(random.random() * 10 ** 8)
        np.random.seed(seed)

        full_dataset = dict()

        for name in self.seasons.keys():
            full_dataset[name] = self.get_injector(name).create_dataset(
                scale, self.get_angular_error_modifier(name)
            )

        return full_dataset

    def trial_function(self, full_dataset):

        llh_functions = dict()
        n_all = dict()

        for name in self.seasons:
            dataset = full_dataset[name]
            llh_f = self.get_likelihood(name).create_llh_function(
                dataset, self.get_angular_error_modifier(name),
                self.make_season_weight
            )
            llh_functions[name] = llh_f
            n_all[name] = len(dataset)

        def f_final(raw_params):

            # If n_s is less than or equal to 0, set gamma to be 3.7 (equal to
            # atmospheric background). This is continuous at n_s=0, but fixes
            # relative weights of sources/seasons for negative n_s values.

            params = list(raw_params)

            if (len(params) > 1) and (params[0] < 0):
                params[1] = 3.7

            # Calculate relative contribution of each source/season

            weights_matrix = self.make_weight_matrix(params)
            weights_matrix /= np.sum(weights_matrix)

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, name in enumerate(self.seasons):
                w = weights_matrix[i][:, np.newaxis]
                ts_val += np.sum(llh_functions[name](params, w))

            return ts_val

        return f_final

    def scan_likelihood(self, scale=0., scan_2d=False):
        """Generic wrapper to perform a likelihood scan a background scramble
        with an injection of signal given by scale.

        :param scale: Flux scale to inject
        """

        res_dict = self.simulate_and_run(scale)

        res = res_dict["res"]
        g = res_dict["f"]

        bounds = list(self.bounds)

        if self.negative_n_s:
            bounds[0] = (-30, 30)

        # Scan 1D Likelihood

        plt.figure(figsize=(8, 4 + 2*len(self.p0)))

        u_ranges = []

        for i, bound in enumerate(bounds):
            ax = plt.subplot(len(self.p0), 1, 1 + i)

            best = list(res.x)
            min_llh = np.sum(float(g(best)))

            factor = 0.9

            if "n_s" in self.param_names[i]:

                best[i] = bound[1]

                while (g(best) > (min_llh + 5.0)):
                    best[i] *= factor

                ur = min(bound[1], max(best[i], 0))

            else:
                ur = bound[1]

            u_ranges.append(ur)

            n_range = np.linspace(float(max(bound[0], -100)), ur, int(1e2))

            # n_range = np.linspace(-30, 30, 1e2)
            y = []

            for n in n_range:

                best[i] = n

                new = g(best)/2.0
                try:
                    y.append(new[0][0])
                except IndexError:
                    y.append(new)

            plt.plot(n_range, y - min(y))
            plt.xlabel(self.param_names[i])
            plt.ylabel(r"$\Delta \log(\mathcal{L}/\mathcal{L}_{0})$")

            logger.info(f"PARAM: {self.param_names[i]}")
            min_y = np.min(y)

            min_index = y.index(min_y)
            min_n = n_range[min_index]

            logger.info(f"Minimum value of {min_y} at {min_n}")

            logger.info("One Sigma interval between")

            l_y = np.array(y[:min_index])
            try:
                l_y = min(l_y[l_y > (min_y + 0.5)])
                l_lim = n_range[y.index(l_y)]
                logger.info(l_lim)
            except ValueError:
                l_lim = min(n_range)
                logger.info(f"<{l_lim}")

            logger.info("and")

            u_y = np.array(y[min_index:])
            try:
                u_y = min(u_y[u_y > (min_y + 0.5)])
                u_lim = n_range[y.index(u_y)]
                logger.info(u_lim)
            except ValueError:
                u_lim = max(n_range)
                logger.info(f">{u_lim}")

            ax.axvspan(l_lim, u_lim, facecolor="grey",
                        alpha=0.2)
            ax.set_ylim(bottom=0.0)

        path = plot_output_dir(self.name) + "llh_scan.pdf"

        title = os.path.basename(
                    os.path.dirname(self.name[:-1])
                ).replace("_", " ") + " Likelihood Scans"

        plt.suptitle(title, y=1.02)

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        plt.savefig(path)
        plt.close()

        logger.info("Saved to {0}".format(path))

        # Scan 2D likelihood

        if np.logical_and(scan_2d, "gamma" in self.param_names):

            gamma_index = self.param_names.index("gamma")

            gamma_bounds = bounds[gamma_index]

            x = np.linspace(gamma_bounds[0], gamma_bounds[1])

            mask = np.array(["n_s" in b for b in self.param_names])

            n_s_bounds = np.array(self.bounds)[mask]

            for j, bound in enumerate(n_s_bounds):
                best = list(res.x)
                plt.figure(figsize=(5.85, 3.6154988341868854))
                ax = plt.subplot(111)

                index = np.arange(len(self.param_names))[mask][j]

                plt.xlabel(r"Spectral Index ($\gamma$)")

                param_name = np.array(self.param_names)[mask][j]
                ylabel = 'n$_{\mathrm{signal}}$' if param_name == 'n_s' else param_name
                plt.ylabel(ylabel)

                y = np.linspace(
                    float(max(bound[0], -100)),
                    np.array(u_ranges)[index],
                    int(1e2)
                )

                X, Y = np.meshgrid(x, y[::-1])
                Z = []

                for gamma in x:
                    best[gamma_index] = gamma
                    z_row = []

                    for n in y:
                        best[index] = n
                        z_row.append((g(best) - g(res.x))/2.0)

                    Z.append(z_row[::-1])

                Z = np.array(Z).T

                levels = 0.5 * np.array([1.0, 2.0, 5.0])**2

                N = 2560
                mmax = np.max(Z)
                mmin = np.min(Z)
                break_ind = int(round(N / (1 + mmax / abs(mmin))))
                top = cm.get_cmap('gray')
                bottom = cm.get_cmap('jet_r', N)
                colorlist = np.empty((N, 4))
                colorlist[break_ind:] = bottom(np.linspace(0, 1, N - break_ind))
                colorlist[:break_ind] = top(np.linspace(1, 0, break_ind))
                cmap = ListedColormap(colorlist)
                norm = Normalize(vmin=mmin, vmax=mmax, clip=True)

                plt.imshow(Z, aspect="auto", cmap=cmap, norm=norm,
                           extent=(x[0], x[-1], y[0], y[-1]),
                           interpolation='bilinear')

                cbar = plt.colorbar()
                CS = ax.contour(X, Y, Z, levels=levels, colors="white")

                fmt = {}
                strs = [r'1$\sigma$', r'2$\sigma$', r'5$\sigma$']
                for l, s in zip(CS.levels, strs):
                    fmt[l] = s

                try:
                    ax.clabel(CS, fmt=fmt, inline=1, fontsize=10, levels=levels,
                              colors="white")
                except TypeError:
                    ax.clabel(CS, levels, fmt=fmt, inline=1, fontsize=10, #levels=levels,
                              colors="white")

                ax.set_xlim((min(x), max(x)))
                ax.set_ylim((min(y), max(y)))

                cbar.set_label(r"$\Delta \log(\mathcal{L}/\mathcal{L}_{0})$",
                               rotation=90)

                path = plot_output_dir(self.name) + (param_name + "_")[4:] + \
                       "contour_scan.pdf"

                title = os.path.basename(
                    os.path.dirname(self.name[:-1])
                ).replace("_", " ") + " Contour Scans"

                plt.scatter(res.x[gamma_index], res.x[index],  color="white",
                            marker="*")

                plt.grid(color="white", linestyle="--", alpha=0.5)

                #plt.suptitle(title)
                plt.tight_layout()
                plt.savefig(path)
                plt.close()

                logger.info("Saved to {0}".format(path))

        return res_dict

    def neutrino_lightcurve(self, seed=None):

        full_dataset = self.prepare_dataset(30., seed)

        for source in self.sources:

            f, (ax0, ax1) = plt.subplots(1, 2,
                                       gridspec_kw={'width_ratios': [19, 1]})

            logE = []
            time = []
            sig = []

            for season in self.seasons:

                # Generate a scrambled dataset, and save it to the datasets
                # dictionary. Loads the llh for the season.
                data = full_dataset[season]
                llh = self.get_likelihood(season)

                mask = llh.select_spatially_coincident_data(data, [source])

                spatial_coincident_data = data[mask]

                t_mask = np.logical_and(
                    np.greater(
                        spatial_coincident_data["time"],
                        llh.sig_time_pdf.sig_t0(source)),
                    np.less(
                        spatial_coincident_data["time"],
                        llh.sig_time_pdf.sig_t1(source))
                )

                coincident_data = spatial_coincident_data[t_mask]

                SoB = llh.estimate_significance(coincident_data, source)

                mask = SoB > 1.

                y = np.log10(SoB[mask])

                if np.sum(mask) > 0:

                    logE += list(10 ** (coincident_data["logE"][mask] - 3))
                    time += list(coincident_data["time"][mask])
                    sig += list(y)

                if llh.sig_time_pdf.sig_t0(source) > llh.sig_time_pdf.t0:

                    ax0.axvline(llh.sig_time_pdf.sig_t0(source), color="k", linestyle="--", alpha=0.5)

                if llh.sig_time_pdf.sig_t1(source) < llh.sig_time_pdf.t1:

                    ax0.axvline(llh.sig_time_pdf.sig_t1(source), color="k", linestyle="--", alpha=0.5)

            cmap = cm.get_cmap('jet')
            norm = mpl.colors.Normalize(vmin=min(logE), vmax=max(logE),
                                        clip=True)
            m = cm.ScalarMappable(norm=norm, cmap=cmap)

            for i, val in enumerate(sig):
                x = time[i]
                ax0.plot([x, x], [0, val], color=m.to_rgba(logE[i]))

            if hasattr(self, "res_dict"):
                params = self.res_dict["Parameters"]
                if len(params) > 1:
                    ax0.axvspan(
                        params[f"t_start ({source['source_name']})"],
                        params[f"t_end ({source['source_name']})"],
                        facecolor="grey",
                        alpha=0.2
                    )
            ax0.set_xlabel("Arrival Time (MJD)")
            ax0.set_ylabel("Log(Signal/Background)")

            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')
            ax1.set_ylabel("Muon Energy Proxy (TeV)")

            ax0.set_ylim(bottom=0)
            # plt.tight_layout()

            path = f"{plot_output_dir(self.name)}neutrino_lightcurve.pdf"

            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass

            logger.info(f"Saving to {path}")

            plt.savefig(path)
            plt.close()

    @staticmethod
    def return_parameter_info(mh_dict):
        params = [[1.], [(0, 1000.)], ["n_s"]]

        params = [
            params[i] + x for i, x in enumerate(
                LLH.get_parameters(mh_dict["llh_dict"])
            )
        ]

        return params[0], params[1], params[2]

    def return_injected_parameters(self, scale):

        n_inj = 0.
        for season_name in self.seasons.keys():
            n_inj += np.sum(self.get_injector(season_name).n_exp["n_exp"] * scale)

        inj_params = {
            "n_s": n_inj
        }
        inj_params.update(LLH.get_injected_parameters(self.mh_dict))

        return inj_params


@MinimisationHandler.register_subclass('large_catalogue')
class LargeCatalogueMinimisationHandler(FixedWeightMinimisationHandler):
    """Class to perform generic minimisations using a 'fixed weights' matrix.
    However, unlike the 'fixed_weight' class, it is optimised for large
    numbers of sources. It uses a custom 'LowMemoryInjector' which is slower
    but much less burdensome for memory.
    """

    compatible_llh = ["standard_matrix"]
    compatible_negative_n_s = False

    def __init__(self, mh_dict):
        FixedWeightMinimisationHandler.__init__(self, mh_dict)

        if self.param_names != ["n_s", "gamma"]:
            raise Exception("{0} parameters are given, when ['n_s','gamma']"
                            "was expected".format(self.param_names))

    def add_injector(self, season, sources):

        if "injector_name" in self.inj_dict.keys():
            if self.inj_dict["injector_name"] not in LargeCatalogueMinimisationHandler.compatible_injectors:
                raise Exception(f"'{self.inj_dict['injector_name']}' was provided as injection_name. "
                                f"Please use any of {LargeCatalogueMinimisationHandler.compatible_injectors}.")
        else:
            self.inj_dict["injector_name"] = "low_memory_injector"

        return season.make_injector(sources, **self.inj_dict)


@MinimisationHandler.register_subclass('fit_weights')
class FitWeightMinimisationHandler(FixedWeightMinimisationHandler):
    compatible_llh = ["spatial", "fixed_energy", "standard"]
    compatible_negative_n_s = False

    def __init__(self, mh_dict):
        FixedWeightMinimisationHandler.__init__(self, mh_dict)

        if self.negative_n_s:
            raise ValueError(
                "Attempted to mix fitting weights with negative n_s.")

    def trial_function(self, full_dataset):

        llh_functions = dict()
        n_all = dict()

        for name in self.seasons:
            dataset = full_dataset[name]
            llh_f = self.get_likelihood(name).create_llh_function(
                dataset, self.get_angular_error_modifier(name),
                self.make_season_weight
            )
            llh_functions[name] = llh_f
            n_all[name] = len(dataset)

        def f_final(params):

            # Creates a matrix fixing the fraction of the total signal that
            # is expected in each Source+Season pair. The matrix is
            # normalised to 1, so that for a given total n_s, the expectation
            # for the ith season for the jth source is given by:
            #  n_exp = n_s * weight_matrix[i][j]

            weights_matrix = self.make_weight_matrix(params)

            for i, row in enumerate(weights_matrix.T):
                if np.sum(row) > 0:
                    row /= np.sum(row)

            # Having created the weight matrix, loops over each season of
            # data and evaluates the TS function for that season

            ts_val = 0
            for i, name in enumerate(self.seasons):
                w = weights_matrix[i][:, np.newaxis]
                ts_val += llh_functions[name](params, w)

            return ts_val

        return f_final

    @staticmethod
    def source_param_name(source):
        return "n_s ({0})".format(source["source_name"])

    @staticmethod
    def return_parameter_info(mh_dict):
        sources = load_catalogue(mh_dict["catalogue"])
        p0 = [1. for _ in sources]
        bounds = [(0., 1000.) for _ in sources]
        names = [FitWeightMinimisationHandler.source_param_name(x)
                 for x in sources]
        params = [p0, bounds, names]

        params = [
            params[i] + x for i, x in enumerate(
                LLH.get_parameters(mh_dict["llh_dict"])
            )
        ]

        return params[0], params[1], params[2]

    def return_injected_parameters(self, scale):

        inj_params = {}

        for source in self.sources:
            name = source["source_name"]
            key = self.source_param_name(source)
            n_inj = 0
            for season_name in self.seasons.keys():
                try:
                    names = [x[0] for x in self.get_injector(season_name).n_exp["source_name"]]
                    if isinstance(names[0], bytes):
                        names = [x.decode() for x in names]

                    if isinstance(name, bytes):
                        name = name.decode()

                    mask = np.array([x == name for x in names])

                    n_inj += np.sum(self.get_injector(season_name).n_exp["n_exp"][mask] * scale)

                # If source not overlapping season, will not be in dict
                except KeyError:
                    pass

            inj_params[key] = n_inj

        inj_params.update(LLH.get_injected_parameters(self.mh_dict))

        return inj_params


@MinimisationHandler.register_subclass("flare")
class FlareMinimisationHandler(FixedWeightMinimisationHandler):

    compatible_llh = ["spatial", "fixed_energy", "standard"]
    compatible_negative_n_s = False

    def __init__(self, mh_dict):
        MinimisationHandler.__init__(self, mh_dict)
        # For each season, we create an independent likelihood, using the
        # source list along with the sets of energy/time
        # PDFs provided in llh_kwargs.
        for name in self.seasons:

            tpdf = self.get_likelihood(name).sig_time_pdf

            # Check to ensure that no weird new untested time PDF is used
            # with the flare search method, since uniform time PDFs over the
            # duration of a given flare is an assumption baked into the PDF
            # construction. New time PDFs could be added, but the Flare class
            #  + LLH would need to be tested first and probably modified.

            if np.sum([isinstance(tpdf, x) for x in [Box, Steady]]) == 0:
                raise ValueError("Attempting to use a time PDF that is not a "
                                 "Box or a Steady time PDF class. The flare "
                                 "search method is only compatible with "
                                 "time PDFs that are uniform over "
                                 "fixed periods.")

    def run_trial(self, full_dataset):

        datasets = dict()

        livetime_calcs = dict()

        time_dict = {
            "time_pdf_name": "custom_source_box"
        }

        results = {
            "Parameters": dict(),
            "Flag": []
        }

        # Loop over each data season

        for (name, season) in self.seasons.items():

            # Generate a scrambled dataset, and save it to the datasets
            # dictionary. Loads the llh for the season.

            data = full_dataset[name]
            llh = self.get_likelihood(name)

            livetime_calcs[name] = TimePDF.create(time_dict, season.get_time_pdf())

            # Loops over each source in catalogue

            for source in self.sources:

                # Identify spatially- and temporally-coincident data

                mask = llh.select_spatially_coincident_data(data, [source])
                spatial_coincident_data = data[mask]

                t_mask = np.logical_and(
                    np.greater(
                        spatial_coincident_data["time"],
                        llh.sig_time_pdf.sig_t0(source)),
                    np.less(
                        spatial_coincident_data["time"],
                        llh.sig_time_pdf.sig_t1(source))
                )

                coincident_data = spatial_coincident_data[t_mask]

                # If there are events in the window...

                if len(coincident_data) > 0:

                    # Creates empty dictionary to save info

                    source_name = source["source_name"]
                    if source_name not in list(datasets.keys()):
                        datasets[source_name] = dict()

                    new_entry = {
                        "season_name": season.season_name
                    }
                    new_entry["Coincident Data"] = coincident_data
                    new_entry["Start (MJD)"] = llh.sig_time_pdf.t0
                    new_entry["End (MJD)"] = llh.sig_time_pdf.t1

                    # Identify significant events (S/B > 1)

                    significant = llh.find_significant_events(
                        coincident_data, source)

                    new_entry["Significant Times"] = significant["time"]

                    new_entry["N_all"] = len(data)

                    datasets[source_name][name] = new_entry

        stacked_ts = 0.0

        # Minimisation of each source

        for (source, source_dict) in datasets.items():

            src = self.sources[self.sources["source_name"] == source][0]
            p0, bounds, names = self.source_fit_parameter_info(self.mh_dict,
                                                               src)

            # Create a full list of all significant times

            all_times = []
            n_tot = 0
            for season_dict in source_dict.values():
                new_times = season_dict["Significant Times"]
                all_times.extend(new_times)
                n_tot += len(season_dict["Coincident Data"])

            all_times = np.array(sorted(all_times))

            # Minimum flare duration (days)
            min_flare = 0.25
            # Conversion to seconds
            min_flare *= 60 * 60 * 24

            # Length of search window in livetime

            search_window = np.sum([
                self.get_likelihood(x).sig_time_pdf.effective_injection_time(src)
                for x in self.seasons.keys()]
            )

            # If a maximum flare length is specified, sets that here

            if "max_flare" in list(self.llh_dict["llh_sig_time_pdf"].keys()):
                # Maximum flare given in days, here converted to seconds
                max_flare = self.llh_dict["llh_sig_time_pdf"]["max_flare"] * (
                        60 * 60 * 24
                )
            else:
                max_flare = search_window

            # Loop over all flares, and check which combinations have a
            # flare length between the maximum and minimum values

            pairs = []

            # print "There are", len(all_times), "significant neutrinos",
            # print "out of", n_tot, "neutrinos"

            for x in all_times:
                for y in all_times:
                    if y > x:
                        pairs.append((x, y))

            # If there is are no pairs meeting this criteria, skip

            if len(pairs) == 0:
                logger.debug("Continuing because no pairs")
                continue

            all_res = []
            all_ts = []
            all_f = []
            all_pairs = []

            # Loop over each possible significant neutrino pair

            for i, pair in enumerate(pairs):
                t_start = pair[0]
                t_end = pair[1]

                # Calculate the length of the neutrino flare in livetime

                flare_time = np.array(
                    (t_start, t_end),
                    dtype=[
                        ("start_time_mjd", np.float),
                        ("end_time_mjd", np.float),
                    ]
                )

                flare_length = np.sum([
                    time_pdf.effective_injection_time(flare_time)
                    for time_pdf in livetime_calcs.values()]
                )

                # If the flare is between the minimum and maximum length

                if flare_length < min_flare:
                    continue
                elif flare_length > max_flare:
                    continue


                # Marginalisation term is length of flare in livetime
                # divided by max flare length in livetime. Accounts
                # for the additional short flares that can be fitted
                # into a given window

                overall_marginalisation = flare_length / max_flare

                # Each flare is evaluated accounting for the
                # background on the sky (the non-coincident
                # data), which is given by the number of
                # neutrinos on the sky during the given
                # flare. (NOTE THAT IT IS NOT EQUAL TO THE
                # NUMBER OF NEUTRINOS IN THE SKY OVER THE
                # ENTIRE SEARCH WINDOW)

                n_all = np.sum([np.sum(~np.logical_or(
                    np.less(data["time"], t_start),
                    np.greater(data["time"], t_end)))
                                for data in full_dataset.values()])

                llhs = dict()

                # Loop over data seasons

                for (name, season_dict) in sorted(source_dict.items()):

                    llh = self.get_likelihood(name)

                    # Check that flare overlaps with season

                    inj_time = llh.sig_time_pdf.effective_injection_time(
                        flare_time
                    )

                    if not inj_time > 0:
                        continue

                    coincident_data = season_dict["Coincident Data"]

                    data = full_dataset[name]

                    n_season = np.sum(~np.logical_or(
                        np.less(data["time"], t_start),
                        np.greater(data["time"], t_end)))

                    # Removes non-coincident data

                    flare_veto = np.logical_or(
                        np.less(coincident_data["time"], t_start),
                        np.greater(coincident_data["time"], t_end)
                    )

                    # Checks to make sure that there are
                    # neutrinos in the sky at all. There should
                    # be, due to the definition of the flare window.

                    if n_all > 0:
                        pass
                    else:
                        raise Exception("Events are leaking somehow!")

                    # Creates the likelihood function for the flare

                    flare_f = llh.create_flare_llh_function(
                        coincident_data, flare_veto, n_all, src, n_season,
                        self.get_angular_error_modifier(season_dict["season_name"])
                    )

                    llhs[season_dict["season_name"]] = {
                        "f": flare_f,
                        "flare length": flare_length
                    }

                # From here, we have normal minimisation behaviour

                def f_final(params):

                    # Marginalisation is done once, not per-season

                    ts = 2 * np.log(overall_marginalisation)

                    for llh_dict in llhs.values():
                        ts += llh_dict["f"](params)

                    return -ts

                res = scipy.optimize.fmin_l_bfgs_b(
                    f_final, p0, bounds=bounds,
                    approx_grad=True)

                all_res.append(res)
                all_ts.append(-res[1])
                all_f.append(f_final)
                all_pairs.append(pair)

            max_ts = max(all_ts)
            stacked_ts += max_ts
            index = all_ts.index(max_ts)

            best_start = all_pairs[index][0]
            best_end = all_pairs[index][1]

            best_time = np.array(
                (best_start, best_end),
                dtype=[
                    ("start_time_mjd", np.float),
                    ("end_time_mjd", np.float),
                ]
            )

            best_length = np.sum([
                time_pdf.effective_injection_time(best_time)
                for time_pdf in livetime_calcs.values()]
            ) / (60 * 60 * 24)

            best = [x for x in all_res[index][0]] + [
                best_start, best_end, best_length
            ]

            p0, bounds, names = self.source_parameter_info(self.mh_dict, src)

            names += [self.source_param_name(x, src)
                      for x in ["t_start", "t_end", "length"]]

            for i, x in enumerate(best):
                key = names[i]
                results["Parameters"][key] = x

            results["Flag"] += [all_res[index][2]["warnflag"]]

            del all_res, all_f, all_times

        results["TS"] = stacked_ts

        del datasets, full_dataset, livetime_calcs

        return results

    @staticmethod
    def source_param_name(param, source):
        return param + " (" + str(source["source_name"]) + ")"


    @staticmethod
    def source_fit_parameter_info(mh_dict, source):

        p0 = [1.]
        bounds = [(0., 1000.)]
        names = [FlareMinimisationHandler.source_param_name("n_s", source)]

        llh_p0, llh_bounds, llh_names = LLH.get_parameters(
            mh_dict["llh_dict"])

        p0 += llh_p0
        bounds += llh_bounds
        names += [FlareMinimisationHandler.source_param_name(x, source)
                  for x in llh_names]

        return p0, bounds, names

    @staticmethod
    def source_parameter_info(mh_dict, source):

        p0, bounds, names = \
            FlareMinimisationHandler.source_fit_parameter_info(
                mh_dict, source
            )

        p0 += [np.nan for _ in range(3)]
        bounds += [(np.nan, np.nan)for _ in range(3)]
        names += [FlareMinimisationHandler.source_param_name(x, source)
                  for x in ["t_start", "t_end", "length"]]

        return p0, bounds, names

    @staticmethod
    def return_parameter_info(mh_dict):
        p0, bounds, names = [], [], []
        sources = load_catalogue(mh_dict["catalogue"])
        for source in sources:
            res = FlareMinimisationHandler.source_parameter_info(
                mh_dict, source
            )

            for i, x in enumerate(res):
                [p0, bounds, names][i] += x

        return p0, bounds, names

    def return_injected_parameters(self, scale):

        inj_params = {}

        for source in self.sources:
            name = source["source_name"]
            key = self.source_param_name("n_s", source)
            n_inj = 0

            for season_name in self.seasons.keys():

                try:

                    names = [x[0] for x in self.get_injector(season_name).n_exp["source_name"]]

                    if isinstance(names[0], bytes):
                        names = [x.decode() for x in names]

                    if isinstance(name, bytes):
                        name = name.decode()

                    mask = np.array([x == name for x in names])

                    n_inj += np.sum(self.get_injector(season_name).n_exp["n_exp"][mask] * scale)

                # If source not overlapping season, will not be in dict
                except KeyError:
                    pass

            inj_params[key] = n_inj

            ts = min([self.get_injector(season_name).sig_time_pdf.sig_t0(source)
                      for season_name in self.seasons.keys()])
            te = max([self.get_injector(season_name).sig_time_pdf.sig_t1(source)
                      for season_name in self.seasons.keys()])

            inj_params[self.source_param_name("length", source)] = te - ts

            if self.time_smear:
                inj_params[self.source_param_name("t_start", source)] = np.nan
                inj_params[self.source_param_name("t_end", source)] = np.nan
            else:
                inj_params[self.source_param_name("t_start", source)] = ts
                inj_params[self.source_param_name("t_end", source)] = te

            for (key, val) in LLH.get_injected_parameters(
                    self.mh_dict).items():
                inj_params[self.source_param_name(key, source)] = val

        return inj_params

    def add_likelihood(self, season):
        return generate_dynamic_flare_class(season, self.sources, self.llh_dict)

if __name__ == '__main__':
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    parser.add_argument("-n", "--n_cpu", default=2)
    cfg = parser.parse_args()

    with open(cfg.file, "rb") as f:
        mh_dict = Pickle.load(f)

    mh = MinimisationHandler.create(mh_dict)

    scales, seeds = mh.trial_params(mh_dict["scale"], n_steps=mh_dict["n_steps"])

    if "fixed_scale" in list(mh_dict.keys()):
        scale = [mh_dict["fixed_scale"] for _ in seeds]
        n_trials = int(float(mh_dict["n_trials"]) / float(cfg.n_cpu))
    else:
        n_trials = int(mh_dict["n_trials"])

    trials = [mh_dict["n_trials"] for _ in seeds]
    loop_args = zip(trials, scales, seeds)

    logger.info("N CPUs:{0}".format(cfg.n_cpu))

    with Pool(int(cfg.n_cpu)) as p:
        p.starmap(mh.run, loop_args)

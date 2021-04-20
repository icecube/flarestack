import logging
import sys
import os
import numpy as np
from flarestack.core.minimisation import MinimisationHandler, read_mh_dict, FlareMinimisationHandler
from flarestack.core.injector import MockUnblindedInjector, \
    TrueUnblindedInjector
from flarestack.core.results import ResultsHandler, OverfluctuationError
from flarestack.core.time_pdf import TimePDF
from flarestack.shared import name_pickle_output_dir, plot_output_dir, \
    analysis_pickle_path, limit_output_path, unblinding_output_path
import pickle
from flarestack.core.ts_distributions import plot_background_ts_distribution, get_ts_fit_type
import matplotlib.pyplot as plt
from flarestack.utils.catalogue_loader import load_catalogue

logger = logging.getLogger(__name__)


def confirm():
    print("Is this correct? (y/n)")

    x = ""

    while x not in ["y", "n"]:
        x = input("")

    if x == "n":
        logger.warning("Please check carefully before unblinding!")
        sys.exit()


def create_unblinder(unblind_dict, mock_unblind=True, full_plots=False,
                     disable_warning=False, reunblind=True, **kwargs):
    """Dynamically create an Unblinder class that inherits corectly from the
    appropriate MinimisationHandler. The name of the parent is specified in
    the unblinder dictionary as 'mh_name'.

    :param unblind_dict: Dictionary containing Unblinding arguments
    :param mock_unblind: By default, the unblinder returns a fixed-seed
    background scramble. mock_unblind must be explicitly set to False for
    unblinding to occur.
    :param full_plots: Boolean, determines whether likelihood scans and
    limits be generated (can be computationally expensive)
    :param disable_warning: By default, the unblinder gives a warning if real
    data is unblinded. This can be disabled.
    :param reunblind: By default, the unblinder performs the analysis and
    save the ns, gamma and TS results in a pickle file. If set to False, the
    pickle file is loaded and unblinding not performed.
    :return: Instance of dynamically-generated Unblinder class
    """

    unblind_dict = read_mh_dict(unblind_dict)

    try:
        mh_name = unblind_dict["mh_name"]
    except KeyError:
        raise KeyError("No MinimisationHandler specified.")

    # Set up dynamic inheritance

    try:
        ParentMiminisationHandler = MinimisationHandler.subclasses[mh_name]
    except KeyError:
        raise KeyError("Parent class {} not found.".format(mh_name))

    # Defines custom Unblinder class

    class Unblinder(ParentMiminisationHandler):

        def __init__(self, unblind_dict, seed=None, scan_2d=False, **kwargs):

            self.unblind_dict = unblind_dict
            unblind_dict["unblind_bool"] = True
            unblind_dict["mock_unblind_bool"] = mock_unblind
            unblind_dict["inj_dict"] = {}
            self.background_median = np.nan

            if np.logical_and(not mock_unblind, not disable_warning):
                self.check_unblind()

            if mock_unblind is False:
                self.mock_unblind = False
            else:
                self.mock_unblind = True

            ParentMiminisationHandler.__init__(self, unblind_dict)


            try:
                if self.mock_unblind:
                    self.limit_path = limit_output_path(
                        self.unblind_dict["name"] + "mock_unblind/")
                    self.unblind_res_path = unblinding_output_path(
                        self.unblind_dict["name"] + "mock_unblind/")

                else:
                    self.limit_path = limit_output_path(
                        self.unblind_dict["name"] + "real_unblind/")
                    self.unblind_res_path = unblinding_output_path(
                        self.unblind_dict["name"] + "real_unblind/")

            except KeyError:
                self.limit_path = np.nan
                self.unblind_res_path = np.nan

            if self.name != " /":
                if self.mock_unblind:
                    self.name += "mock_unblind/"
                else:
                    self.name += "real_unblind/"


            self.plot_dir = plot_output_dir(self.name)

            try:
                os.makedirs(os.path.dirname(self.unblind_res_path))
            except OSError:
                pass

            # Avoid redoing unblinding

            if not reunblind:

                try:
                    logger.info("Not re-unblinding, loading results from {0}".format(self.unblind_res_path))

                    with open(self.unblind_res_path, "rb") as f:
                        self.res_dict = pickle.load(f)

                except FileNotFoundError:
                    logger.warning("No pickle files containing unblinding results found. "
                                    "Re-unblinding now.")

            # Otherwise unblind
            
            if not hasattr(self, "res_dict"):
                logger.info("Unblinding catalogue")

                # Minimise likelihood and produce likelihood scans
                self.res_dict = self.simulate_and_run(0, seed)

            logger.info(self.res_dict)

            # Quantify the TS value significance
            self.ts = np.array([self.res_dict["TS"]])[0]
            self.sigma = np.nan
            self.ts_type = get_ts_fit_type(unblind_dict)

            logger.info("Test Statistic of: {0}".format(self.ts))

            ub_res_dict = {
                "Parameters": self.res_dict['Parameters'],
                "TS": self.res_dict['TS'],
                "Flag": self.res_dict['Flag']
            }

            logger.info("Saving unblinding results to {0}".format(self.unblind_res_path))

            with open(self.unblind_res_path, "wb") as f:
                pickle.dump(ub_res_dict, f)

            try:
                path = self.unblind_dict["background_ts"]
                self.pickle_dir = name_pickle_output_dir(path)
                self.output_file = self.plot_dir + "TS.pdf"
                self.compare_to_background_TS()
            except KeyError:
                logger.warning("No Background TS Distribution specified. "
                                "Cannot assess significance of TS value.")


            if full_plots:

                self.calculate_upper_limits()
                if isinstance(self, FlareMinimisationHandler):
                    self.neutrino_lightcurve()
                else:
                    self.scan_likelihood(scan_2d=scan_2d)

        def add_injector(self, season, sources):
            if self.mock_unblind is False:
                return TrueUnblindedInjector(
                    season, sources, **self.inj_dict)
            else:
                return MockUnblindedInjector(
                    season, sources, **self.inj_dict)

        def calculate_upper_limits(self):

            try:
                ul_dir = os.path.join(self.plot_dir, "upper_limits/")

                try:
                    os.makedirs(ul_dir)
                except OSError:
                    pass

                flux_uls = []
                fluence_uls = []
                e_per_source_uls = []
                energy_flux = []
                x_axis = []

                for subdir in sorted(os.listdir(self.pickle_dir)):

                    root = os.path.join(self.unblind_dict["background_ts"], subdir)

                    new_path = analysis_pickle_path(name=root)

                    logger.info(f'Opening file {new_path}')

                    with open(new_path, "rb") as f:
                        mh_dict = pickle.load(f)
                        e_pdf_dict = mh_dict["inj_dict"]["injection_energy_pdf"]

                    self.unblind_dict['name'] = mh_dict['name']
                    rh = ResultsHandler(self.unblind_dict)

                    logger.debug(f"In calculate_upper_limits, ResultsHandler is {rh}")

                    savepath = ul_dir + subdir + ".pdf"

                    if self.ts < self.bkg_median:
                        logging.warning(f"Specified TS ({self.ts}) less than the background median ({self.bkg_median}). "
                                        f"There was thus a background underfluctuation. "
                                        f"Using the sensitivity for an upper limit.")

                    ul, extrapolated, err = rh.find_overfluctuations(max(self.ts, self.bkg_median), savepath)

                    flux_uls.append(ul)

                    # Calculate mean injection time per source

                    n_sources = float(len(self.sources))

                    inj_time = 0.

                    seasons = mh_dict["dataset"]
                    for (name, season) in seasons.items():
                        t_pdf = TimePDF.create(
                            mh_dict["inj_dict"]["injection_sig_time_pdf"], season.get_time_pdf()
                        )
                        for src in self.sources:
                            inj_time += t_pdf.raw_injection_time(src) / \
                                        n_sources

                    astro_dict = rh.nu_astronomy(ul, e_pdf_dict)

                    key = "Energy Flux (GeV cm^{-2} s^{-1})"

                    fluence_uls.append(
                        astro_dict[key]
                        * inj_time
                    )

                    e_per_source_uls.append(
                        astro_dict["Mean Luminosity (erg/s)"] * inj_time
                    )

                    energy_flux.append(astro_dict[key])
                    x_axis.append(float(subdir))

                plt.figure()
                ax = plt.subplot(111)
                plt.plot(x_axis, flux_uls, marker = 'o', label="upper limit")

                if self.mock_unblind:
                    ax.text(0.2, 0.5, "MOCK DATA", color="grey", alpha=0.5, transform=ax.transAxes)

                ax.set_xlabel('Gamma')
                ax.set_ylabel('Flux')
                plt.yscale("log")
                plt.savefig(self.plot_dir + "upper_limit_flux.pdf")
                plt.close()

                plt.figure()
                ax1 = plt.subplot(111)
                ax2 = ax1.twinx()

                ax1.plot(x_axis, fluence_uls, marker='o')
                ax2.plot(x_axis, e_per_source_uls, marker='o')

                if self.mock_unblind:
                    ax1.text(0.2, 0.5, "MOCK DATA", color="grey", alpha=0.5, transform=ax1.transAxes)

                ax2.grid(True, which='both')
                ax1.set_xlabel('Gamma')
                ax2.set_xlabel('Gamma')
                ax1.set_ylabel(r"Total Fluence [GeV cm$^{-2}$ s$^{-1}$]")
                ax2.set_ylabel(r"Isotropic-Equivalent Luminosity $L_{\nu}$ (erg/s)")
                ax1.set_yscale("log")
                ax2.set_yscale("log")

                for k, ax in enumerate([ax1, ax2]):
                    y = [fluence_uls, e_per_source_uls][k]
                    ax.set_ylim(0.95 * min(y), 1.1 * max(y))

                plt.tight_layout()
                plt.savefig(self.plot_dir + "upper_limit_fluence.pdf")
                plt.close()

                try:
                    os.makedirs(os.path.dirname(self.limit_path))
                except OSError:
                    pass
                logger.info("Saving limits to {0}".format(self.limit_path))

                res_dict = {
                    "gamma": x_axis,
                    "flux": flux_uls,
                    "energy_flux": energy_flux,
                    "fluence": fluence_uls,
                    "energy": e_per_source_uls
                }

                with open(self.limit_path, "wb") as f:
                    pickle.dump(res_dict, f)

            except AttributeError as e:
                logger.warning("Unable to set limits. No TS distributions found.")

            except ValueError as e:
                raise OverfluctuationError("Insufficent pseudotrial values above and below threshold")

        def compare_to_background_TS(self):
            logger.info("Retrieving Background TS Distribution from {0}".format(self.pickle_dir))

            try:

                ts_array = list()

                for subdir in os.listdir(self.pickle_dir):

                    merged_pkl = os.path.join(os.path.join(self.pickle_dir, subdir), "merged/0.pkl")

                    logger.info("Loading {0}".format(merged_pkl))

                    with open(merged_pkl, 'rb') as mp:

                        merged_data = pickle.load(mp)

                    ts_array += list(merged_data["TS"])

                ts_array = np.array(ts_array)

                self.sigma = plot_background_ts_distribution(
                    ts_array, self.output_file, self.ts_type, self.ts, self.mock_unblind)

                self.background_median = np.median(ts_array)

            except (IOError, OSError):

                try:
                    ts_array = list()

                    merged_pkl = os.path.join(self.pickle_dir, "merged/0.pkl")

                    logger.info("No subfolders found, loading {0}".format(merged_pkl))

                    with open(merged_pkl, 'rb') as mp:
                        merged_data = pickle.load(mp)

                    ts_array += list(merged_data["TS"])

                    ts_array = np.array(ts_array)

                    self.sigma = plot_background_ts_distribution(
                        ts_array, self.output_file, self.ts_type, self.ts, self.mock_unblind)

                except (IOError, OSError):
                    logger.warning("No Background TS Distribution found")
                    pass

        def check_unblind(self):
            print("\n")
            print("You are proposing to unblind data.")
            print("\n")
            confirm()
            print("\n")
            print("You are unblinding the following catalogue:")
            print("\n")
            print(self.unblind_dict["catalogue"])
            print("\n")
            confirm()
            print("\n")
            print("The catalogue has the following entries:")
            print("\n")

            cat = load_catalogue(self.unblind_dict["catalogue"])

            print(cat.dtype.names)
            print(len(cat), np.sum(cat['base_weight']))
            print(cat)
            print("\n")
            confirm()
            print("\n")
            print("The following datasets will be used:")
            print("\n")
            for x in self.unblind_dict["dataset"].values():
                print(x.sample_name, x.season_name)
                print("\n")
                print(x.exp_path)
                print(x.pseudo_mc_path)
                print("\n")
            confirm()
            print("\n")
            print("The following LLH will be used:")
            print("\n")
            for (key, val) in self.unblind_dict["llh_dict"].items():
                print(key, val)
            print("\n")
            confirm()
            print("\n")
            print("Are you really REALLY sure about this?")
            print("You will unblind. This is your final warning.")
            print("\n")
            confirm()
            print("\n")
            print("OK, you asked for it...")
            print("\n")

    return Unblinder(unblind_dict, **kwargs)

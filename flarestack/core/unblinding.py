import sys
import os
import numpy as np
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.results import ResultsHandler
from flarestack.core.time_PDFs import TimePDF
from flarestack.shared import name_pickle_output_dir, plot_output_dir, \
    analysis_pickle_path, limit_output_path
import cPickle as Pickle
from flarestack.core.ts_distributions import plot_background_ts_distribution
import matplotlib.pyplot as plt


def confirm():
    print "Is this correct? (y/n)"

    x = ""

    while x not in ["y", "n"]:
        x = raw_input("")

    if x == "n":
        print "\n"
        print "Please check carefully before unblinding!"
        print "\n"
        sys.exit()


class Unblinder(MinimisationHandler):

    def __init__(self, unblind_dict, mock_unblind=True, full_plots=False):
        self.unblind_dict = unblind_dict
        unblind_dict["Unblind"] = True
        unblind_dict["Mock Unblind"] = mock_unblind
        unblind_dict["inj kwargs"] = {}

        if not mock_unblind:
            self.check_unblind()
        MinimisationHandler.__init__(self, unblind_dict)

        if self.mock_unblind:
            self.name += "mock_unblind/"
            self.limit_path = limit_output_path(
                self.unblind_dict["background TS"] + "mock_unblind/")
        else:
            self.name += "real_unblind/"
            self.limit_path = limit_output_path(
                self.unblind_dict["background TS"] + "real_unblind/")

        self.plot_dir = plot_output_dir(self.name)

        # Minimise likelihood and produce likelihood scans
        self.res_dict = self.run_trial(0)

        print "\n"
        print self.res_dict
        print "\n"

        # Quantify the TS value significance
        # print type(np.array([self.res_dict["TS"]]))
        self.ts = np.array([self.res_dict["TS"]])[0]
        self.sigma = np.nan

        print "Test Statistic of:", self.ts

        if self.fit_weights:
            self.ts_type = "Fit Weights"
        elif self.flare:
            self.ts_type = "Flare"
        else:
            self.ts_type = "Standard"

        try:
            path = self.unblind_dict["background TS"]
            self.pickle_dir = name_pickle_output_dir(path)
            self.output_file = self.plot_dir + "TS.pdf"
            self.compare_to_background_TS()
        except KeyError:
            print "No Background TS Distribution specified.",
            print "Cannot assess significance of TS value."

        if full_plots:

            self.calculate_upper_limits()

            if self.flare:
                self.neutrino_lightcurve()
            else:
                self.scan_likelihood()

    def calculate_upper_limits(self):

        ul_dir = self.plot_dir + "upper_limits/"

        try:
            os.makedirs(ul_dir)
        except OSError:
            pass

        flux_uls = []
        fluence_uls = []
        e_per_source_uls = []
        x_axis = []

        for subdir in os.listdir(self.pickle_dir):
            new_path = self.unblind_dict["background TS"] + subdir + "/"

            with open(analysis_pickle_path(new_path), "r") as f:
                mh_dict = Pickle.load(f)
                e_pdf_dict = mh_dict["inj kwargs"]["Injection Energy PDF"]

            rh = ResultsHandler(new_path, self.unblind_dict["llh kwargs"],
                                self.unblind_dict["catalogue"])

            savepath = ul_dir + subdir + ".pdf"

            ul, extrapolated = rh.set_upper_limit(float(self.ts), savepath)
            flux_uls.append(ul)

            # Calculate mean injection time per source

            n_sources = float(len(self.sources))

            inj_time = 0.

            for season in mh_dict["datasets"]:

                t_pdf = TimePDF.create(
                    mh_dict["inj kwargs"]["Injection Time PDF"], season
                )

                for src in self.sources:
                    inj_time += t_pdf.raw_injection_time(src)/n_sources

            astro_dict = rh.nu_astronomy(ul, e_pdf_dict)

            fluence_uls.append(
                astro_dict["Total Fluence (GeV cm^{-2} s^{-1})"] * inj_time)

            e_per_source_uls.append(
                astro_dict["Mean Luminosity (erg/s)"] * inj_time
            )

            x_axis.append(float(subdir))

        plt.figure()
        plt.plot(x_axis, flux_uls, label="upper limit")
        plt.yscale("log")
        plt.savefig(self.plot_dir + "upper_limit_flux.pdf")
        plt.close()

        plt.figure()
        ax1 = plt.subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(x_axis, fluence_uls)
        ax2.plot(x_axis, e_per_source_uls)

        ax2.grid(True, which='both')
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
        print "Saving limits to", self.limit_path

        res_dict = {
            "x": x_axis,
            "flux": flux_uls,
            "fluence": fluence_uls,
            "energy": e_per_source_uls
        }

        with open(self.limit_path, "wb") as f:
            Pickle.dump(res_dict, f)

    def compare_to_background_TS(self):
        print "Retrieving Background TS Distribution from", self.pickle_dir

        try:

            ts_array = list()

            for subdir in os.listdir(self.pickle_dir):
                merged_pkl = self.pickle_dir + subdir + "/merged/0.pkl"

                print "Loading", merged_pkl

                with open(merged_pkl) as mp:

                    merged_data = Pickle.load(mp)

                ts_array += list(merged_data["TS"])

            ts_array = np.array(ts_array)

            self.sigma = plot_background_ts_distribution(
                ts_array, self.output_file, self.ts_type, self.ts)

        except IOError, OSError:
            print "No Background TS Distribution found"
            pass

    def check_unblind(self):
        print "\n"
        print "You are proposing to unblind data."
        print "\n"
        confirm()
        print "\n"
        print "You are unblinding the following catalogue:"
        print "\n"
        print self.unblind_dict["catalogue"]
        print "\n"
        confirm()
        print "\n"
        print "The catalogue has the following entries:"
        print "\n"

        cat = np.load(self.unblind_dict["catalogue"])

        print cat.dtype.names
        print cat
        print "\n"
        confirm()
        print "\n"
        print "The following datasets will be used:"
        print "\n"
        for x in self.unblind_dict["datasets"]:
            print x["Data Sample"], x["Name"]
            print "\n"
            print x["exp_path"]
            print x["mc_path"]
            print x["grl_path"]
            print "\n"
        confirm()
        print "\n"
        print "The following LLH will be used:"
        print "\n"
        for (key, val) in self.unblind_dict["llh kwargs"].iteritems():
            print key, val
        print "\n"
        confirm()
        print "\n"
        print "Are you really REALLY sure about this?"
        print "You will unblind. This is your final warning."
        print "\n"
        confirm()
        print "\n"
        print "OK, you asked for it..."
        print "\n"

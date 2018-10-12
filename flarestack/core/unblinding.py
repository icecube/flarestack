import sys
import os
import numpy as np
from flarestack.core.minimisation import MinimisationHandler
from flarestack.shared import name_pickle_output_dir, plot_output_dir
import cPickle as Pickle
from flarestack.core.ts_distributions import plot_background_ts_distribution


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

    def __init__(self, unblind_dict, mock_unblind=True):
        self.unblind_dict = unblind_dict
        unblind_dict["Unblind"] = True
        unblind_dict["Mock Unblind"] = mock_unblind
        unblind_dict["inj kwargs"] = {}

        if not mock_unblind:
            self.check_unblind()
        MinimisationHandler.__init__(self, unblind_dict)

        # Minimise likelihood and produce likelihood scans
        self.res_dict = self.run_trial(0)

        print "\n"
        print self.res_dict
        print "\n"

        # Quantify the TS value significance
        # print type(np.array([self.res_dict["TS"]]))
        self.ts = np.array([self.res_dict["TS"]])[0]

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
            self.output_file = plot_output_dir(self.name) + "TS.pdf"
            self.compare_to_background_TS()
        except KeyError:
            print "No Background TS Distribution specified.",
            print "Cannot assess significance of TS value."

        if self.flare:
            self.neutrino_lightcurve()
        else:
            self.scan_likelihood()

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

            plot_background_ts_distribution(ts_array, self.output_file,
                                            self.ts_type, self.ts)

        except IOError:
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

import sys
import os
import numpy as np
from core.minimisation import MinimisationHandler
from shared import name_pickle_output_dir, plot_output_dir
import cPickle as Pickle
from core.ts_distributions import plot_background_ts_distribution


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

    def __init__(self, unblind_dict):
        self.unblind_dict = unblind_dict
        unblind_dict["Mock Unblind"] = True
        unblind_dict["inj kwargs"] = {}

        # self.check_unblind()
        MinimisationHandler.__init__(self, unblind_dict)

        # Minimise likelihood and produce likelihood scans
        self.res = self.scan_likelihood()

        # Quantify the TS value significance
        self.ts = -self.res["fun"] * np.sign(self.res["x"][0])

        print "Test Statistic of:", self.ts

        if self.fit_weights:
            self.ts_type = "Fit Weights"
        elif self.flare:
            self.ts_type = "Flare"
        else:
            self.ts_type = "Standard"

        try:
            path = self.unblind_dict["background TS"]
            self.merged_dir = name_pickle_output_dir(path) + "merged/0.pkl"
            self.output_file = plot_output_dir(self.name) + "TS.pdf"
            self.compare_to_background_TS()
        except KeyError:
            print "No Background TS Distribution specified.",
            print "Cannot assess significance of TS value."

    def compare_to_background_TS(self):
        print "Retrieving Background TS Distribution from", self.merged_dir

        with open(self.merged_dir) as mp:

            merged_data = Pickle.load(mp)

        ts_array = merged_data["TS"]

        plot_background_ts_distribution(ts_array, self.output_file,
                                        self.ts_type, self.ts)

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

import sys
import numpy as np
from core.minimisation import MinimisationHandler

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
        self.check_unblind()
        MinimisationHandler.__init__(self, unblind_dict)

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
        sys.exit()

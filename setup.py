# create FlareStack__Data
# eval cvmfs ...
# create virtualenv
# Create Output Folder for plots
import os
import sys
from config import scratch_path
from shared import fs_scratch_dir, input_dir, storage_dir, output_dir, \
    log_dir, catalogue_dir, acc_f_dir, pickle_dir, plots_dir
from utils.prepare_catalogue import make_single_sources
from utils.create_acceptance_functions import make_acceptance_f

if __name__ == "__main__":
    print "\n \n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*                Initialising setup for FlareStack                 *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    print "  Initialising directory for data storage. This could be a scratch  "
    print "                   space or local directory.                        "
    print "\n"

    print "The following parent directory has been found in config.py: \n"
    print "\t", scratch_path
    print
    print "A new data storage directory will be created at: \n"
    print "\t", fs_scratch_dir
    print
    print "Is this correct? (y/n)"

    x = ""

    while x not in ["y", "n"]:
        x = raw_input("")

    if x == "n":
        print "\n"
        print "Please edit config.py to include the correct directory!"
        print "\n"
        sys.exit()

    for dir in [input_dir, storage_dir, output_dir, log_dir, catalogue_dir,
                acc_f_dir, pickle_dir, plots_dir]:
        if not os.path.isdir(dir):
            print "Making Directory:", dir
            os.makedirs(dir)
        else:
            print "Found Directory:", dir

    print "\n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*                 Initialising catalogue creation                  *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    make_single_sources()

    print "\n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*                   Making Acceptance Functions                    *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    make_acceptance_f()

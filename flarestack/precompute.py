import os
import sys
import argparse
import numpy as np

config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.py"

# Scratch directory can be changed if needed

def set_scratch_directory(path):
    """Sets the scratch directory to be a custom path, and exports this.

    :param path: Path to scratch
    """
    
    if path[-1] != "/":
        path += "/"

    if not os.path.isdir(path):
        raise Exception("Attempting to set invalid path for scratch. "
                        "Directory", path, "does not exist!")
    print "Setting scratch path to", path

    with open(config_path, "wb") as f:
        f.write("scratch_path = '" + path + "'")


def run_precompute(all_data):
    """Builds directory substructure, creates standard source catalogues and
    creates acceptance functions + Signal/Background splines

    :param all_data: All datasets to be used for setup
    """
    import flarestack.config
    from flarestack.shared import fs_scratch_dir, input_dir, storage_dir, \
        output_dir, \
        log_dir, catalogue_dir, acc_f_dir, pickle_dir, plots_dir, \
        SoB_spline_dir, analysis_dir, illustration_dir, \
        transients_dir, bkg_spline_dir, dataset_dir
    from flarestack.utils.prepare_catalogue import make_single_sources
    from flarestack.utils.create_acceptance_functions import make_acceptance_f
    from flarestack.utils.make_SoB_splines import make_spline

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
    print "\t", flarestack.config.scratch_path
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
        return

    for dirname in [input_dir, storage_dir, output_dir, log_dir, catalogue_dir,
                    acc_f_dir, pickle_dir, plots_dir,
                    SoB_spline_dir, analysis_dir, illustration_dir,
                    transients_dir, bkg_spline_dir]:
        if not os.path.isdir(dirname):
            print "Making Directory:", dirname
            os.makedirs(dirname)
        else:
            print "Found Directory:", dirname

    print "\n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*                 Initialising catalogue creation                  *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    make_single_sources()

    # Check to ensure there is at least one IceCube dataset present

    x = np.sum([os.path.isdir(os.path.dirname(y["mc_path"])) for y in all_data])

    print "********************************************************************"
    print "*                                                                  *"
    print "*                     Checking data directories                    *"
    print "*                                                                  *"
    print "********************************************************************"

    roots = list(set([os.path.dirname(y["mc_path"]) for y in all_data]))

    if x == 0:
        print "No IceCube data files found. Tried searching for: \n"
        for y in roots:
            print "\t", y

        print ""
        print "Download these data files yourself, and save them to: \n"
        print "\t", dataset_dir
        print "\n"
        sys.exit()

    else:
        print "Searched for the following directories: \n"
        for y in roots:
            print "\t", y,
            print "Found?", os.path.isdir(y)

    print "\n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*                   Making Acceptance Functions                    *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    make_acceptance_f(all_data)

    print "\n"
    print "********************************************************************"
    print "*                                                                  *"
    print "*    Creating Log(Energy) vs. Sin(Declination) Sig/Bkg splines     *"
    print "*                                                                  *"
    print "********************************************************************"
    print "\n"
    make_spline(all_data)


if __name__ == "__main__":

    with open(config_path, "r") as f:
        scratch_path = f.readline()[16:-2]

    parser = argparse.ArgumentParser()

    parser.add_argument("-sp", "--scratch_path",
                        help="Path to scratch directory",
                        default=scratch_path)

    cfg = parser.parse_args()

    if cfg.scratch_path != scratch_path:
        del scratch_path
        set_scratch_directory(cfg.scratch_path)

    from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1

    icecube_data = txs_sample_v1

    run_precompute(icecube_data)

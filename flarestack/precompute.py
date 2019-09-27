import os
import argparse
import numpy as np

# Scratch directory can be changed if needed

def run_precompute(ask=True):
    """Builds directory substructure, creates standard source catalogues and
    creates acceptance functions + Signal/Background splines

    :param all_data: All datasets to be used for setup
    :param ask: Explicitly ask before running precompute
    """
    import flarestack.config
    from flarestack.shared import fs_scratch_dir, all_dirs
    from flarestack.utils.prepare_catalogue import make_single_sources
    from flarestack.utils.create_acceptance_functions import make_acceptance_f
    from flarestack.utils.make_SoB_splines import make_spline
    from flarestack.icecube_utils.dataset_loader import verify_grl_with_data
    import socket
    from flarestack.cluster.make_desy_cluster_script import \
        make_desy_submit_file

    print("\n \n")
    print("********************************************************************")
    print("*                                                                  *")
    print("*                Initialising setup for FlareStack                 *")
    print("*                                                                  *")
    print("********************************************************************")
    print("\n")
    print("  Initialising directory for data storage. This could be a scratch  ")
    print("                   space or local directory.                        ")
    print("\n")

    print("The following parent directory has been found in config.py: \n")
    print("\t", flarestack.config.scratch_path)
    print()
    print("A new data storage directory will be created at: \n")
    print("\t", fs_scratch_dir)
    print()

    if ask:
        print("Is this correct? (y/n)")

        x = ""

        while x not in ["y", "n"]:
            x = input("")

        if x == "n":
            print("\n")
            print("Please edit config.py to include the correct directory!")
            print("\n")
            return

    for dirname in all_dirs:
        if not os.path.isdir(dirname):
            print("Making Directory:", dirname)
            os.makedirs(dirname)
        else:
            print("Found Directory:", dirname)

    host = socket.gethostname()

    print("\n")
    print("********************************************************************")
    print("*                                                                  *")
    print("*                    Generating Cluster Scripts                    *")
    print("*                                                                  *")
    print("********************************************************************")
    print("\n")

    if np.logical_or("ifh.de" in host, "zeuthen.desy.de" in host):
        make_desy_submit_file()
    else:
        print("Host", host, "not recognised.")
        print("No Cluster Scripts generated.")

    # print("********************************************************************")
    # print("*                                                                  *")
    # print("*                     Checking data directories                    *")
    # print("*                                                                  *")
    # print("********************************************************************")
    #
    # if not isinstance(all_data, list):
    #     all_data = [all_data]
    #
    # for dataset in all_data:
    #     for y in dataset:
    #         y.check_files_exist()

    # print("\n")
    # print("********************************************************************")
    # print("*                                                                  *")
    # print("*                       Checking GoodRunLists                      *")
    # print("*                                                                  *")
    # print("********************************************************************")
    # print("\n")
    # verify_grl_with_data(all_data)
    #
    # print("\n")
    # print("********************************************************************")
    # print("*                                                                  *")
    # print("*                   Making Acceptance Functions                    *")
    # print("*                                                                  *")
    # print("********************************************************************")
    # print("\n")
    # make_acceptance_f(all_data)
    #
    # print("\n")
    # print("********************************************************************")
    # print("*                                                                  *")
    # print("*    Creating Log(Energy) vs. Sin(Declination) Sig/Bkg splines     *")
    # print("*                                                                  *")
    # print("********************************************************************")
    # print("\n")
    # make_spline(all_data)


# if __name__ == "__main__":
#
#     with open(config_path, "r") as f:
#         scratch_path = f.readline()[16:-2]
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("-sp", "--scratch_path",
#                         help="Path to scratch directory",
#                         default=scratch_path)
#
#     cfg = parser.parse_args()
#
#     if cfg.scratch_path != scratch_path:
#         del scratch_path
#         set_scratch_directory(cfg.scratch_path)
#
#     from flarestack.data.public import icecube_ps_3_year
#
#     run_precompute(icecube_ps_3_year)

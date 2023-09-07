""" Script for making band masks for each season when 'large_catalogue' minimizer is chosen.
    In case jobs are submitted in the cluster, this script is ran locally first,
    performing 1 trial in order to write the band masks per catalogue,
    which are then loaded when running trials on the cluster.
    This tackles issue #7 in flarestack repo
"""


from flarestack.core.minimisation import MinimisationHandler


import logging

logger = logging.getLogger("flarestack.main")


def make_band_mask(mh_dict):
    """Initialize the injector (ie the LowMemoryInjector)
    and write the band masks for each catalog chunk in the catalogue_cache dir
    This works only with the LargeCatalogueMinimisationHandler
    """

    assert mh_dict["mh_name"] == "large_catalogue", "mh_name != large_catalogue"

    # only 1 trial for writing the band masks
    mh_dict["n_trials"] = 1

    mh = MinimisationHandler.create(mh_dict)

    for season in mh.seasons.keys():
        inj = mh.get_injector(season)
        inj.make_injection_band_mask()



if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    args = parser.parse_args()

    logger.info("Making band masks for the catalogue")

    with open(args.file, "rb") as f:
        mh_dict = pickle.load(f)

    make_band_mask(mh_dict)

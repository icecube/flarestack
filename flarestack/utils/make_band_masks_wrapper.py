""" Script for making band masks for each season when 'large_catalogue' minimizer is chosen.
    In case jobs are submitted in the cluster, this script is ran locally first,
    performing 1 trial in order to write the band masks per catalogue,
    which are then loaded when running trials on the cluster.
    This tackles issue #7 in flarestack repo
"""

import os
import numpy as np
from flarestack.core.minimisation import MinimisationHandler
from flarestack.shared import band_mask_cache_name
from flarestack.utils.catalogue_loader import load_catalogue


import logging

logger = logging.getLogger(__name__)


def make_band_mask(mh_dict):
    """Initialize the injector (ie the LowMemoryInjector)
    and write the band masks for each catalog chunk in the catalogue_cache dir
    This works only with the LargeCatalogueMinimisationHandler
    """

    assert mh_dict["mh_name"] == "large_catalogue", "mh_name != large_catalogue"

    logger.debug("Run 1 trial locally to make band masks for the catalog")

    # only 1 trial for writing the band masks
    mh_dict["n_trials"] = 1

    mh = MinimisationHandler.create(mh_dict)

    # injection_declination_bandwidth defaults to 1.5 when MCInjector is initialized
    if "injection_declination_bandwidth" not in mh.inj_dict.keys():
        logger.info(
            "Injection declination bandwidth not specified, setting default value 1.5"
        )
        mh.inj_dict["injection_declination_bandwidth"] = 1.5

    injection_bandwidth = mh.inj_dict["injection_declination_bandwidth"]

    # check if band masks are already written for the catalog
    sources = load_catalogue(mh_dict["catalogue"])

    for i in mh.seasons.values():
        cats, paths, mask_indices, source_indices = band_mask_cache_name(
            i, sources, injection_bandwidth
        )

        if np.sum([not os.path.isfile(x) for x in paths]) > 0.0:
            logger.info(
                "No band masks found for season {0}, running 1 trial to write them".format(
                    i.season_name
                )
            )
            mh.get_injector(i.season_name)
            # inj = mh.get_injector(i.season_name)
            # inj.make_injection_band_mask()
        else:
            logger.info("Band masks exist for season {0}".format(i.season_name))
            continue


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        mh_dict = pickle.load(f)

    make_band_mask(mh_dict)

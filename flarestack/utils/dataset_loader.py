import numpy as np
from numpy.lib.recfunctions import append_fields, rename_fields
from flarestack.shared import min_angular_err


def data_loader(data_path):
    """Helper function to load data for a given season/set of season.
    Adds sinDec field if this is not available, and combines multiple years
    of data is appropriate (different sets of data from the same icecube
    configuration should be given as a list)

    :param data_path: Path to data or list of paths to data
    :return: Loaded Dataset (experimental or MC)
    """

    if isinstance(data_path, list):
        dataset = np.concatenate(
            tuple([np.load(x) for x in data_path]))
    else:
        dataset = np.load(data_path)

    if "sinDec" not in dataset.dtype.names:

        new_dtype = np.dtype([("sinDec", np.float)])

        sinDec = np.array(np.sin(dataset["dec"]), dtype=new_dtype)

        dataset = append_fields(
            dataset, 'sinDec', sinDec, usemask=False, dtypes=[np.float]
        )

    if "sigma" not in dataset.dtype.names:

        if "angErr" in dataset.dtype.names:
            dataset = rename_fields(dataset, {"angErr": "sigma"})
        else:
            raise Exception("No recognised Angular Error field found in "
                            "dataset. (Searched for 'sigma' and 'angErr')")

    # Apply a minimum angular error "floor"

    dataset["sigma"][dataset["sigma"] < min_angular_err] = min_angular_err

    return dataset


def grl_loader(season):
    if isinstance(season["grl_path"], list):
        grl = np.sort(np.array(np.concatenate(
            [np.load(x) for x in season["grl_path"]])),
            order="run")
    else:
        grl = np.load(season["grl_path"])

    if np.sum(~grl["good_i3"]) == 0:
        pass
    else:
        print "Trying to load", season
        print "The following runs are included:"
        print grl[~grl["good_i3"]]
        raise Exception("Runs marked as 'bad' are found in Good Run List")

    if "length" not in grl.dtype.names:

        if "livetime" in grl.dtype.names:
            grl = rename_fields(grl, {"livetime": "length"})
        else:
            raise Exception("No recognised Livetime field found in "
                            "GoodRunList. (Searched for 'livetime' and "
                            "'length')")

    return grl

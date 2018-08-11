import numpy as np
from numpy.lib.recfunctions import append_fields


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

    return dataset

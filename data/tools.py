import numpy as np
from numpy.lib.recfunctions import append_fields


def data_loader(datapath):
    """Helper function to load icecube data for a given season/set of season.
    Adds sinDec field if this is not available, and combines multiple years
    of data is appropriate (different sets of data from the same icecube
    configuration should be given as a list)

    :param datapath: Path to data or list of paths to data
    :return: Loaded Dataset (experimental or MC)
    """

    if isinstance(datapath, list):
        exp = np.concatenate(
            tuple([np.load(x) for x in datapath]))
    else:
        exp = np.load(datapath)

    if "sinDec" not in exp.dtype.names:

        new_dtype = np.dtype([("sinDec", np.float)])

        sinDec = np.array(np.sin(exp["dec"]), dtype=new_dtype)

        exp = append_fields(exp, 'sinDec', sinDec, usemask=False, dtypes=[
            np.float])

    return exp

import numpy as np
from numpy.lib.recfunctions import append_fields


def data_loader(datapath):

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

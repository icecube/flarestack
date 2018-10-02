"""Script to produce catalogues for use in stacking analysis.

The catalogues themselves are randomly produced for the purpose of trialing
the code. Modification of variable n can produces a catalogue with an
arbitrary number of sources.

"""

import numpy as np
import time
import os
from sys import stdout
from flarestack.shared import catalogue_dir

cat_dtype = [
    ("ra", np.float), ("dec", np.float),
    ("Relative Injection Weight", np.float),
    ("Ref Time (MJD)", np.float),
    ("Start Time (MJD)", np.float),
    ("End Time (MJD)", np.float),
    ('Distance (Mpc)', np.float), ('Name', 'a30'),
]


def single_source(sindec):
    """Produces a catalogue with a single source_path.

    :param sindec: Sin(Declination) of Source
    :return: Source Array
    """
    sources = np.empty(
        1, dtype=cat_dtype)

    ref_time = 55800.4164699

    sources['ra'] = np.array([np.deg2rad(180.)])
    sources['dec'] = np.arcsin(sindec)
    sources['Relative Injection Weight'] = np.array([1.])
    sources['Distance (Mpc)'] = np.array([1.0])
    sources['Ref Time (MJD)'] = (np.array([ref_time]))
    sources['Start Time (MJD)'] = (np.array([ref_time - 50]))
    sources['End Time (MJD)'] = (np.array([ref_time + 100]))
    sources['Name'] = 'PS_dec=' + str(sindec)

    return sources


def ps_catalogue_name(sindec):
    return catalogue_dir + "single_source/sindec_" + '{0:.2f}'.format(sindec)\
           + ".npy"


def make_single_sources():
    """Makes single-source catalogues for a variety of sindec intervals."""
    print "Making single-source catalogues for the following sin(declinations):"

    sindecs = np.linspace(1.00, -1.00, 41)
    print sindecs, "\n"

    try:
        os.makedirs(os.path.dirname(ps_catalogue_name(0.0)))
    except OSError:
        pass

    for sindec in sindecs:
        cat = single_source(sindec)
        save_path = ps_catalogue_name(sindec)
        stdout.write("\rSaving to " + save_path)
        stdout.flush()
        np.save(save_path, cat)
        time.sleep(0.1)

    print "\n"
    print "Single Source catalogues created!", "\n"


def custom_sources(name, ra, dec, weight, distance, ref_time=np.nan,
                   start_time=np.nan, end_time=np.nan):
    """Creates a catalogue array,

    :param name: Source Name
    :param ra: Right Ascension (Degrees)
    :param dec: Declination (Degrees)
    :param weight: Relative Weights for Source Injection
    :param distance: Distance to source (a.u.)
    :param ref_time: Reference Time (MJD)
    :param start_time: Start Time for window (MJD)
    :param end_time: End Time for window (MJD)

    :return: Catalogue Array
    """
    sources = np.empty(np.array([ra]).__len__(), dtype=cat_dtype)

    sources['ra'] = np.array([np.deg2rad(ra)])
    sources['dec'] = np.deg2rad(np.array([dec]))

    # If some sources are to be brighter than others, a non-uniform weight
    # array can be passed. This array is normalised, such that the mean
    # weight is 1.
    sources['Relative Injection Weight'] = np.array([weight]) * float(len(
        np.array([weight])))/np.sum(weight)

    # The source distance can be provided, in arbitrary units. The injector
    # and reconstructor will weight sources according to 1/ (distance ^ 2).

    sources['Distance (Mpc)'] = np.array([distance])

    # The source reference time can be arbitrarily defined, for example as
    # the discovery date or the date of lightcurve peak. It is important that
    # this is consistently defined between sources. Box Time PDFs can be defined
    # relative to this point.
    sources['Ref Time (MJD)'] = (np.array([ref_time]))

    # The source csan also be assigned fixed start and end times. Fixed Box
    # Time PDFs can be defined relative to these values. This allows for the
    # Time PDF duration to vary between sources.
    sources['Start Time (MJD)'] = (np.array([start_time]))
    sources['End Time (MJD)'] = (np.array([end_time]))

    sources['Name'] = np.array([name])

    return sources


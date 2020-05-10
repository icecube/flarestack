"""Script to produce catalogues for use in stacking analysis.

The catalogues themselves are randomly produced for the purpose of trialing
the code. Modification of variable n can produces a catalogue with an
arbitrary number of sources.

"""
import numpy as np
import os
import logging
import random
import zlib
from flarestack.shared import catalogue_dir

cat_dtype = [
    ("ra_rad", np.float), ("dec_rad", np.float),
    ("base_weight", np.float),
    ("injection_weight_modifier", np.float),
    ("ref_time_mjd", np.float),
    ("start_time_mjd", np.float),
    ("end_time_mjd", np.float),
    ('distance_mpc', np.float), ('source_name', 'a30'),
]


def single_source(sindec, ra_rad=np.pi):
    """Produces a catalogue with a single source_path.

    :param sindec: Sin(Declination) of Source
    :param ra: Right Ascension in radians
    :return: Source Array
    """
    sources = np.empty(
        1, dtype=cat_dtype)

    ref_time = 55800.4164699

    sources['ra_rad'] = np.array([ra_rad])
    sources['dec_rad'] = np.arcsin(sindec)
    sources['base_weight'] = np.array([1.])
    sources['injection_weight_modifier'] = np.array([1.])
    sources['distance_mpc'] = np.array([1.0])
    sources['ref_time_mjd'] = (np.array([ref_time]))
    sources['start_time_mjd'] = (np.array([ref_time - 50]))
    sources['end_time_mjd'] = (np.array([ref_time + 100]))
    sources['source_name'] = 'PS_dec=' + str(sindec)

    return sources


def build_ps_cat_name(sindec):
    return catalogue_dir + "single_source/sindec_" + '{0:.2f}'.format(sindec)\
           + ".npy"

def build_ps_stack_cat_name(sindecs):
    return f"{catalogue_dir}multi_source/{zlib.adler32(str(list(sindecs)).encode())}.npy"

def make_single_source(sindec):
    cat = single_source(sindec)
    save_path = build_ps_cat_name(sindec)
    try:
        os.makedirs(os.path.dirname(save_path))
    except FileExistsError:
        pass
    logging.info("Saving to {0}".format(save_path))
    np.save(save_path, cat)


def ps_catalogue_name(sindec):
    name = build_ps_cat_name(sindec)
    
    if not os.path.isfile(name):
        make_single_source(sindec)

    return name

def make_stacked_source(sindecs):
    cat = []

    for sindec in sindecs:

        ra_rad = random.random() ** 2 * np.pi

        cat.append(single_source(sindec, ra_rad=ra_rad))

    cat = np.array(cat, dtype=cat[0].dtype).T[0]

    save_path = build_ps_stack_cat_name(sindecs)

    try:
        os.makedirs(os.path.dirname(save_path))
    except FileExistsError:
        pass

    logging.info("Saving to {0}".format(save_path))
    np.save(save_path, cat)


def ps_stack_catalogue_name(*args):

    name = build_ps_stack_cat_name(args)

    if not os.path.isfile(name):
        make_stacked_source(args)

    return name


def make_single_sources():
    """Makes single-source catalogues for a variety of sindec intervals."""
    logging.info("Making single-source catalogues for the following sin(declinations):")

    sindecs = np.linspace(1.00, -1.00, 41)
    logging.info(sindecs)

    try:
        os.makedirs(os.path.dirname(ps_catalogue_name(0.0)))
    except OSError:
        pass

    for sindec in sindecs:
        make_single_source(sindec)

    logging.info("Single Source catalogues created!")


def custom_sources(name, ra, dec, weight, distance,
                   injection_modifier=None, ref_time=np.nan,
                   start_time=np.nan, end_time=np.nan):
    """Creates a catalogue array,

    :param name: Source Name
    :param ra: Right Ascension (Degrees)
    :param dec: Declination (Degrees)
    :param weight: Relative Weights
    :param distance: Distance to source (a.u.)
    :param ref_time: Reference Time (MJD)
    :param start_time: Start Time for window (MJD)
    :param end_time: End Time for window (MJD)

    :return: Catalogue Array
    """
    sources = np.empty(np.array([ra]).__len__(), dtype=cat_dtype)

    sources['ra_rad'] = np.deg2rad(np.array([ra]))
    sources['dec_rad'] = np.deg2rad(np.array([dec]))

    # If some sources are to be brighter than others, a non-uniform weight
    # array can be passed.
    sources['base_weight'] = np.array([weight])

    # The source distance can be provided, in arbitrary units. The injector
    # and reconstructor will weight sources according to 1/ (distance ^ 2).

    sources['distance_mpc'] = np.array([distance])

    # The sources can have a modified injection weight. This means the
    # weights used in the likelihood will not match the weights used in the
    # injection stage

    if injection_modifier is not None:
        sources["injection_weight_modifier"] = np.array(injection_modifier)
    else:
        sources["injection_weight_modifier"] = np.ones_like(ra)

    # The source reference time can be arbitrarily defined, for example as
    # the discovery date or the date of lightcurve peak. It is important that
    # this is consistently defined between sources. Box Time PDFs can be defined
    # relative to this point.
    sources['ref_time_mjd'] = (np.array([ref_time]))

    # The source csan also be assigned fixed start and end times. Fixed Box
    # Time PDFs can be defined relative to these values. This allows for the
    # Time PDF duration to vary between sources.
    sources['start_time_mjd'] = (np.array([start_time]))
    sources['end_time_mjd'] = (np.array([end_time]))

    sources['source_name'] = np.array([name])

    return sources


"""Script to produce catalogues for use in stacking analysis.

The catalogues themselves are randomly produced for the purpose of trialing
the code. Modification of variable n can produces a catalogue with an
arbitrary number of sources.

"""

import numpy as np
import time
from sys import stdout
from common import cat_path

def read_in_catalogue():
    """Produces a catalogue with a single source_path.

    :return: Source Array
    """
    sources = np.empty(
        1, dtype=[("ra", np.float), ("dec", np.float),
                  ("flux", np.float), ("n_exp", np.float),
                  ("weight", np.float), ("weight_acceptance", np.float),
                  ("weight_time", np.float),
                  ("weight_distance", np.float),
                  ("discoverydate_mjd", np.float),
                  ("distance", np.float), ('name', 'a30'),
                  ])

    sources['ra'] = np.array([np.deg2rad(180.)])
    sources['dec'] = np.arcsin(-0.95)
    sources['flux'] = np.array([1.e-9])
    sources['weight'] = np.array([1.0])
    sources['distance'] = np.array([1.0])
    sources['discoverydate_mjd'] = (
        np.array([55800.4164699]) )
    sources['name'] = 'SN_01'
    sources["n_exp"] = 0.0
    sources["weight_acceptance"] = 0.0
    sources["weight_time"] = 0.0
    sources["weight_distance"] = 0.0

    return sources


def single_source(sindec):
    """Produces a catalogue with a single source_path.

    :param sindec: Sin(Declination) of Source
    :return: Source Array
    """
    sources = np.empty(
        1, dtype=[("ra", np.float), ("dec", np.float),
                  ("flux", np.float), ("n_exp", np.float),
                  ("weight", np.float), ("weight_acceptance", np.float),
                  ("weight_time", np.float),
                  ("weight_distance", np.float),
                  ("discoverydate_mjd", np.float),
                  ("distance", np.float), ('name', 'a30'),
                  ])

    sources['ra'] = np.array([np.deg2rad(180.)])
    sources['dec'] = np.arcsin(sindec)
    sources['flux'] = np.array([1.e-9])
    sources['weight'] = np.array([1.0])
    sources['distance'] = np.array([1.0])
    sources['discoverydate_mjd'] = (
        np.array([55800.4164699]) )
    sources['name'] = 'PS_dec=' + str(sindec)
    sources["n_exp"] = 0.0
    sources["weight_acceptance"] = 0.0
    sources["weight_time"] = 0.0
    sources["weight_distance"] = 0.0

    return sources

def make_single_sources():
    """Makes single-source catalogues for a variety of sindec intervals."""
    print "Making single-source catalogues for the following sin(declinations):"

    sindecs = np.linspace(1.00, -1.00, 41)
    print sindecs, "\n"
    save_name = cat_path + "single_source_dec_"

    for sindec in sindecs:
        cat = single_source(sindec)
        save_path = save_name + '{0:.2f}'.format(sindec) + \
                    ".npy"
        stdout.write("\rSaving to " + save_path)
        stdout.flush()
        np.save(save_path, cat)
        time.sleep(0.1)

    print "\n"
    print "Single Source catalogues created!", "\n"

def read_in_catalogue_stack(n_sources):
    """Produces a catalogue of n sources. Attributes are randomised within
    physical bounds.

    :param n_sources: Number of sources in catalogue
    :return: Source Array
    """

    sources = np.empty(
        n_sources, dtype=[("ra", np.float), ("dec", np.float),
                  ("flux", np.float), ("n_exp", np.float),
                  ("weight", np.float), ("weight_acceptance", np.float),
                  ("weight_time", np.float),
                  ("weight_distance", np.float),
                  ("norm_time", np.float),
                  ("global_weight_norm_time", np.float),
                  ("discoverydate_mjd", np.float),
                  ("distance", np.float), ('name', 'a30'),
                  ])

    sources['ra'] = np.deg2rad(np.linspace(0., 360, n_sources + 1)[:-1])
    sources['dec'] = np.deg2rad(np.linspace(-90., 90., n_sources + 2)[1:-1])

    sources['flux'] = np.ones_like(sources['ra'])
    normalisation = n_sources * 1.e-9 / np.sum(sources['flux'])
    sources['flux'] *= normalisation
    sources['weight'] = np.ones_like(sources['ra'])
    sources['distance'] = np.ones_like(sources['ra'])
    sources['discoverydate_mjd'] = (
        55694.4164699 + (np.array(range(n_sources)) / float(n_sources)) *
        368.00423609999416)
    sources['name'] = ['SN' + str(i) for i in range(n_sources)]

    return sources

if __name__ == '__main__':
    
    # # Saves a single-source_path catalogue
    # single_source_array = read_in_catalogue()
    # np.save(cat_path + "catalogue00.npy", single_source_array)
    #
    # # Saves an n-source_path catalogue
    # n = 10
    # n_source_array = read_in_catalogue_stack(n)
    # np.save(cat_path + "catalogue_stack" + str(n) + ".npy", n_source_array)

    make_single_sources()

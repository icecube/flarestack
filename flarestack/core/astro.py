"""
Function taken from IceCube astro package.
"""

import numexpr
import numpy as np


def angular_distance(lon1, lat1, lon2, lat2):
    """
    calculate the angular distince along the great circle
    on the surface of a shpere between the points
    (`lon1`,`lat1`) and (`lon2`,`lat2`)
    This function Works for equatorial coordinates
    with right ascension as longitude and declination
    as latitude. This function uses the Vincenty formula
    for calculating the distance.
    Parameters
    ----------
    lon1 : array_like
      longitude of first point in radians
    lat1 : array_like
      latitude of the first point in radians
    lon2 : array_like
      longitude of second point in radians
    lat2 : array_like
      latitude of the second point in radians
    """
    c1 = np.cos(lat1)
    c2 = np.cos(lat2)
    s1 = np.sin(lat1)
    s2 = np.sin(lat2)
    sd = np.sin(lon2 - lon1)
    cd = np.cos(lon2 - lon1)

    return np.arctan2(np.hypot(c2 * sd, c1 * s2 - s1 * c2 * cd), s1 * s2 + c1 * c2 * cd)


# Fast angular distance using numexpr to avoid creating large temporary arrays
fast_angular_distance = numexpr.NumExpr(
    "arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(fmod(lon2 - lon1 + pi, 2 * pi) - pi))",
    signature=[
        ("lon1", np.float64),
        ("lat1", np.float64),
        ("lon2", np.float64),
        ("lat2", np.float64),
        ("pi", np.float64),
    ],
)

# NB: numexpr implements fmod but not mod, and mod is equivalent to abs(fmod(...))
in_ra_window = numexpr.NumExpr(
    "abs(abs(fmod(lon1 - lon2 + pi, 2 * pi)) - pi) < dPhi",
    signature=[
        ("lon1", np.float64),
        ("lon2", np.float64),
        ("pi", np.float64),
        ("dPhi", np.float64),
    ],
)

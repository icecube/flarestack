import numpy as np
from numpy.lib.recfunctions import append_fields, rename_fields


def distance_scaled_weight(sources: np.ndarray) -> np.ndarray:
    return sources["base_weight"] * sources["distance_mpc"] ** -2


def distance_scaled_weight_sum(cls, sources: np.ndarray) -> float:
    return np.sum(distance_scaled_weight(sources))


def load_catalogue(path):
    sources = np.load(path)

    # Maintain backwards-compatibility

    maps = [
        ("ra", "ra_rad"),
        ("dec", "dec_rad"),
        ("Relative Injection Weight", "injection_weight_modifier"),
        ("Distance (Mpc)", "distance_mpc"),
        ("Name", "source_name"),
        ("Ref Time (MJD)", "ref_time_mjd"),
        ("Start Time (MJD)", "start_time_mjd"),
        ("End Time (MJD)", "end_time_mjd"),
    ]

    for old_key, new_key in maps:
        if old_key in sources.dtype.names:
            sources = rename_fields(sources, {old_key: new_key})

    if "base_weight" not in sources.dtype.names:
        base_weight = np.ones(len(sources))

        sources = append_fields(
            sources, "base_weight", base_weight, usemask=False, dtypes=[float]
        )

    # Check that ra and dec are really in radians!

    if max(sources["ra_rad"]) > 2.0 * np.pi:
        raise Exception(
            "Sources have Right Ascension values greater than 2 "
            "pi. Are you sure you're not using degrees rather "
            "than radians?"
        )

    if max(abs(sources["dec_rad"])) > np.pi / 2.0:
        raise Exception(
            "Sources have Declination values exceeding "
            "+/- pi/2. Are you sure you're not using degrees "
            "rather than radians?"
        )

    # Check that all sources have a unique name
    if len(set(sources["source_name"])) < len(sources["source_name"]):
        raise Exception(
            "Some sources in catalogue do not have unique "
            "names. Please assign unique names to each source."
        )

    # TODO: add a check on `injection_weight_modifier`

    # Order sources by distance
    sources = np.sort(sources, order="distance_mpc")

    return sources

import numpy as np
import logging
from astropy.table import Table, vstack
from numpy.lib.recfunctions import rename_fields
from flarestack.shared import min_angular_err
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_FIELDS = {
    "time",
    "ra",
    "dec",
    "sigma",
    "angErr",
    "logE",
    "trueE",
    "trueRa",
    "trueDec",
    "ow",
    "sinDec",
    "raw_sigma",
}


def _load_data_table(path_or_str, cut_fields=True) -> Table:
    path = Path(path_or_str)
    if path.with_suffix(".npz").exists():
        data = np.load(path.with_suffix(".npz"))
        fields = data.keys()
        copy = False
    else:
        data = np.load(path, allow_pickle=True)
        fields = data.dtype.names
        # Copy fields into individual columns to improve memory locality
        copy = True
    if cut_fields:
        return Table({k: data[k] for k in fields if k in ALLOWED_FIELDS}, copy=copy)
    else:
        return Table(data, copy=copy)


def data_loader(data_path, floor=True, cut_fields=True) -> Table:
    """Helper function to load data for a given season/set of season.
    Adds sinDec field if this is not available, and combines multiple years
    of data is appropriate (different sets of data from the same icecube
    configuration should be given as a list)

    :param data_path: Path to data or list of paths to data
    :param cut_fields: Boolean to remove unused fields from datasets on loading
    :return: Loaded Dataset (experimental or MC)
    """

    if isinstance(data_path, list):
        dataset = vstack(tuple(_load_data_table(x, cut_fields) for x in data_path))
    else:
        dataset = _load_data_table(data_path, cut_fields)

    if "sinDec" not in dataset.columns:
        dataset.add_column(np.sin(dataset["dec"]), name="sinDec")

    # Check if 'run' or 'Run'

    if "run" not in dataset.columns:
        if "Run" in dataset.columns:
            dataset.rename_column("Run", "run")

    # Check if 'sigma' or 'angErr' is Used

    if "sigma" not in dataset.columns:
        if "angErr" in dataset.columns:
            dataset.rename_column("angErr", "sigma")
        else:
            raise Exception(
                "No recognised Angular Error field found in "
                "dataset. (Searched for 'sigma' and 'angErr')"
            )

    if "raw_sigma" not in dataset.columns:
        dataset.add_column(dataset["sigma"], name="raw_sigma")

    # Apply a minimum angular error "floor"
    if floor:
        dataset["sigma"][dataset["sigma"] < min_angular_err] = min_angular_err

    # prevent accidental in-place updates
    for col in dataset.columns.values():
        col.setflags(write=False)

    return dataset


def grl_loader(season):
    if isinstance(season.grl_path, list):
        grl = np.sort(
            np.array(np.concatenate([np.load(x) for x in season.grl_path])), order="run"
        )
    else:
        grl = np.load(season.grl_path)

    # Check if bad runs are found in GRL
    try:
        if np.sum(~grl["good_i3"]) == 0:
            pass
        else:
            logger.error("Trying to load", season)
            logger.error("The following runs are included:")
            logger.error(grl[~grl["good_i3"]])
            raise Exception("Runs marked as 'bad' are found in Good Run List")
    except ValueError:
        # It seems now good_i3 field is deprecated?
        logger.debug(
            "No field called 'good_i3' found in GoodRunList."
            "Assuming that all runs in GoodRunList are actually good."
        )

    if "length" not in grl.dtype.names:
        if "livetime" in grl.dtype.names:
            grl = rename_fields(grl, {"livetime": "length"})
        else:
            raise Exception(
                "No recognised Livetime field found in "
                "GoodRunList. (Searched for 'livetime' and "
                "'length')"
            )

    # Check if there are events in runs not found in GRL

    exp_data = season.get_exp_data()
    if "run" in exp_data.dtype.names:
        bad_runs = [x for x in set(exp_data["run"]) if x not in grl["run"]]
        if len(bad_runs) > 0:
            raise Exception(
                "Trying to use GoodRunList, but events in data have "
                "runs that are not included on this GoodRunList. \n"
                "Please check to make sure both the GoodRunList, "
                "and the event selection, are correct. \n"
                + "The following runs are affected: \n"
                + str(bad_runs)
            )

    # Sometimes, inexplicable, the runs come in random orders rather than
    # ascending order. This deals with that.

    grl = np.sort(grl, order="run")

    del exp_data

    return grl


def verify_grl_with_data(seasons):
    logger.info(
        "Verifying that, for each dataset, all events are in runs that \n"
        "are on the GRL, and not outside the period marked as good in the "
        "GRL."
    )

    for name, season in seasons.items():
        print(name)

        exp_data = season.get_exp_data(cut_fields=False)

        grl = season.get_grl()

        # Check if there are events in runs that are on the GRL, but outside the
        # period marked as good in the GRL

        n_overflow = 0.0
        affected_runs = []

        for run in grl:
            data = exp_data[exp_data["run"] == run["run"]]
            mask = np.logical_and(
                data["time"] >= run["start"], data["time"] <= run["stop"]
            )

            if np.sum(~mask) > 0:
                n_overflow += np.sum(~mask)
                affected_runs.append(run["run"])

        if n_overflow > 0.0:
            fraction = float(n_overflow) / float(len(exp_data))

            raise Exception(
                "Found events in data set "
                + season["Name"]
                + " which are in runs from the GoodRunList, \n but "
                "the times of these runs lie outside the periods "
                "marked as good. \n In total, "
                + str(fraction)
                + " of events are affected. \n The following runs are"
                " affected: \n" + str(affected_runs)
            )

        else:
            print("Passed!")

        del exp_data

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

    # Check if 'run' or 'Run'

    if "run" not in dataset.dtype.names:

        if "Run" in dataset.dtype.names:
            dataset = rename_fields(dataset, {"Run": "run"})
        else:
            print "\n \n Warning, no run information found. Will not be able " \
                  "to verify that runs are good! \n \n"

    # Check if 'sigma' or 'angErr' is Used

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

    # Check if bad runs are found in GRL

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



    # Check if there are events in runs not found in GRL

    exp_data = data_loader(season["exp_path"])
    if "run" in exp_data.dtype.names:
        bad_runs = [x for x in set(exp_data["run"]) if x not in grl["run"]]

        if len(bad_runs) > 0:
            raise Exception("Trying to use GoodRunList, but events in data have "
                            "runs that are not included on this GoodRunList. \n"
                            "Please check to make sure both the GoodRunList, "
                            "and the event selection, are correct. \n" +
                            "The following runs are affected: \n" +
                            str(bad_runs))

    del exp_data

    return grl


def verify_grl_with_data(datasets):

    print "Verifying that, for each dataset, all events are in runs that \n" \
          "are on the GRL, and not outside the period marked as good in the " \
          "GRL."

    for season in datasets:
        print season["Name"],

        exp_data = data_loader(season["exp_path"])

        grl = grl_loader(season)

        # Check if there are events in runs that are on the GRL, but outside the
        # period marked as good in the GRL

        n_overflow = 0.
        affected_runs = []

        for run in grl:
            data = exp_data[exp_data["run"] == run["run"]]
            mask = np.logical_and(data["time"] >= run["start"],
                                  data["time"] <= run["stop"])

            if np.sum(~mask) > 0:
                n_overflow += np.sum(~mask)
                affected_runs.append(run["run"])

        if n_overflow > 0.:

            fraction = float(n_overflow)/float(len(exp_data))

            raise Exception("Found events in data set " + season["Name"] +
                            " which are in runs from the GoodRunList, \n but "
                            "the times of these runs lie outside the periods "
                            "marked as good. \n In total, " + str(fraction) +
                            " of events are affected. \n The following runs are"
                            " affected: \n" + str(affected_runs))

        else:
            print "Passed!"

        del exp_data

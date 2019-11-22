import numpy as np
from numpy.lib.recfunctions import append_fields, rename_fields
from flarestack.shared import min_angular_err
from scipy.interpolate import interp1d

def data_loader(data_path, floor=True, cut_fields=True):
    """Helper function to load data for a given season/set of season.
    Adds sinDec field if this is not available, and combines multiple years
    of data is appropriate (different sets of data from the same icecube
    configuration should be given as a list)

    :param data_path: Path to data or list of paths to data
    :param cut_fields: Boolean to remove unused fields from datasets on loading
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

    # Check if 'sigma' or 'angErr' is Used

    if "sigma" not in dataset.dtype.names:

        if "angErr" in dataset.dtype.names:
            dataset = rename_fields(dataset, {"angErr": "sigma"})
        else:
            raise Exception("No recognised Angular Error field found in "
                            "dataset. (Searched for 'sigma' and 'angErr')")

    dataset = append_fields(
        dataset, 'raw_sigma', dataset["sigma"], usemask=False, dtypes=[np.float]
    )

    # Apply a minimum angular error "floor"
    if floor:
        dataset["sigma"][dataset["sigma"] < min_angular_err] = min_angular_err

    if cut_fields:

        allowed_fields = ['time', 'ra', 'dec', 'sigma', 'logE', 'trueE',
                          'trueRa', 'trueDec', 'ow', 'sinDec', 'raw_sigma']

        mask = [x for x in dataset.dtype.names if x in allowed_fields]

        dataset = dataset[mask]

    return dataset


def grl_loader(season):

    if isinstance(season.grl_path, list):
        grl = np.sort(np.array(np.concatenate(
            [np.load(x) for x in season.grl_path])),
            order="run")
    else:
        grl = np.load(season.grl_path)
        
    # Check if bad runs are found in GRL
    try:
        if np.sum(~grl["good_i3"]) == 0:
            pass
        else:
            print("Trying to load", season)
            print("The following runs are included:")
            print(grl[~grl["good_i3"]])
            raise Exception("Runs marked as 'bad' are found in Good Run List")
    except ValueError:
        print("No field called 'good_i3'. Cannot check GoodRunList.")

    if "length" not in grl.dtype.names:

        if "livetime" in grl.dtype.names:
            grl = rename_fields(grl, {"livetime": "length"})
        else:
            raise Exception("No recognised Livetime field found in "
                            "GoodRunList. (Searched for 'livetime' and "
                            "'length')")

    # Check if there are events in runs not found in GRL

    exp_data = season.get_exp_data()
    if "run" in exp_data.dtype.names:
        bad_runs = [x for x in set(exp_data["run"]) if x not in grl["run"]]
        if len(bad_runs) > 0:
            raise Exception("Trying to use GoodRunList, but events in data have "
                            "runs that are not included on this GoodRunList. \n"
                            "Please check to make sure both the GoodRunList, "
                            "and the event selection, are correct. \n" +
                            "The following runs are affected: \n" +
                            str(bad_runs))

    # Sometimes, inexplicable, the runs come in random orders rather than
    # ascending order. This deals with that.

    grl = np.sort(grl, order="run")

    del exp_data

    return grl


def convert_grl(season):
    grl = season.get_grl()

    t0 = min(grl["start"])
    t1 = max(grl["stop"])

    full_livetime = np.sum(grl["length"])

    step = 1e-10

    t_range = [t0 - step]
    f = [0.]

    mjd = [0.]
    livetime = [0.]
    total_t = 0.

    for i, run in enumerate(grl):
        mjd.append(run["start"])
        livetime.append(total_t)
        total_t += run["length"]
        mjd.append(run["stop"])
        livetime.append(total_t)

        t_range.extend([
            run["start"] - step, run["start"], run["stop"],
            run["stop"] + step
        ])
        f.extend([0., 1., 1., 0.])

    stitch_t = [t_range[0]]
    stitch_f = [1.]
    for i, t in enumerate(t_range[1:]):
        gap = t - t_range[i - 1]

        if gap < 1e-5 and f[i] == 0:
            pass
        else:
            stitch_t.append(t)
            stitch_f.append(f[i])

    if stitch_t != sorted(stitch_t):
        print("Error in ordering GoodRunList!")
        print("Runs are out of order!")

        print(grl[:5])
        input("prompt")

        for j, t in enumerate(stitch_t):
            if t != sorted(stitch_t)[j]:
                print(j, t, grl[j])
        input("prompt")

    mjd.append(1e5)
    livetime.append(total_t)

    season_f = interp1d(stitch_t, stitch_f, kind="linear")
    mjd_to_livetime = interp1d(mjd, livetime, kind="linear")
    livetime_to_mjd = interp1d(livetime, mjd, kind="linear")

    return t0, t1, full_livetime, season_f, mjd_to_livetime, livetime_to_mjd


def verify_grl_with_data(seasons):

    print("Verifying that, for each dataset, all events are in runs that \n" \
          "are on the GRL, and not outside the period marked as good in the " \
          "GRL.")

    for name, season in seasons.items():
        print(name)

        exp_data = season.get_exp_data(cut_fields=False)

        grl = season.get_grl()

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
            print("Passed!")

        del exp_data

import numpy as np
import os
import cPickle as Pickle
from scipy import interpolate
from data.icecube_pointsource_7_year import ps_dict
from shared import livetime_dir, illustration_dir


def make_grl_f(data_sample):

    sample_name = data_sample["Data Sample"]

    for season in data_sample["Seasons"]:

        new_dict = dict(season)

        grl = np.load(season["grl_path"])

        if np.sum(~grl["good_i3"]) == 0:
            pass
        else:
            raise Exception("Runs marked as 'bad' are found in Good Run List")

        start = min(grl["start"])
        end = max(grl["stop"])

        new_dict["Start Time (MJD)"] = start
        new_dict["End Time (MJD)"] = end
        new_dict["Livetime"] = np.sum(grl["length"])

        t_range = [start]
        f = [0.]

        for run in grl:
            t_range.append(run["start"] - 1e-9)
            t_range.append(run["start"])
            t_range.append(run["stop"])
            t_range.append(run["stop"] + 1e-9)
            f.extend([0., 1., 1., 0.])

        livetime_f = interpolate.interp1d(t_range, f, kind="linear")

        print max(livetime_f(t_range))

        root_dir = livetime_dir + sample_name + "/"

        try:
            os.makedirs(root_dir)
        except OSError:
            pass

        savepath = root_dir + season["Name"] + "_livetime.npy"

        # with open(savepath, "wb") as f:
        #     Pickle.dump(livetime_f, f)
        #
        # t = np.linspace(start, start + 50., 10**4.)[1:-2]
        #
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        # ax = plt.subplot(111)
        # ax.fill_between(t, 1., livetime_f(t))
        # plt.savefig(illustration_dir + season["Name"] + "_livetime.pdf")
        # plt.close()

        print new_dict

make_grl_f(ps_dict)
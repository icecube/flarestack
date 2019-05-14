import os
import numpy as np
import csv
import pickle
from flarestack.shared import public_dataset_dir
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.utils.dataset_loader import data_loader
ref_data = data_loader(IC86_1_dict["exp_path"])

src_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

data_dir = src_dir + "raw_data/"
output_base_dir = public_dataset_dir + "all_sky_3_year/"
output_data_dir = output_base_dir + "events/"


for path in [output_data_dir]:

    try:
        os.makedirs(path)
    except OSError:
        pass


def data_path(season):
    return output_data_dir + season + ".npy"


data_dtype = np.dtype([
    ('ra', np.float),
    ('dec', np.float),
    ('logE', np.float),
    ('sigma', np.float),
    ('time', np.float),
    ('sinDec', np.float)
])


datasets = ["IC79-2010", "IC86-2011", "IC86-2012"]


def parse_numpy_dataset():
    """Function to parse the .txt file  of events into a numpy format
    readable by flarestack, which is the saved in the products/ subdirectory.
    """

    for dataset in datasets:

        data = []

        path = data_dir + dataset + "-events.txt"

        with open(path, "r") as f:

            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):
                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    entry = (np.deg2rad(row[3]), np.deg2rad(row[4]),
                             row[1], np.deg2rad(row[2]),
                             row[0], np.sin(np.deg2rad(row[4]))
                             )

                    data.append(entry)

        data = np.array(data, dtype=data_dtype)

        exp_path = data_path(dataset)

        with open(exp_path, "wb") as f:
            print("Saving converted numpy array to", exp_path)
            pickle.dump(data, f)


def make_season(season_name):
    season_dict = {
        "Data Sample": "all_sky_3_year",
        "sinDec bins": np.unique(np.concatenate([
            np.linspace(-1., -0.9, 2 + 1),
            np.linspace(-0.9, -0.2, 8 + 1),
            np.linspace(-0.2, 0.2, 15 + 1),
            np.linspace(0.2, 0.9, 12 + 1),
            np.linspace(0.9, 1., 2 + 1),
        ])),
        "MJD Time Key": "time",
        "Name": season_name,
        "exp_path": data_path(season_name),
        "mc_path": None,
        "grl_path": None
    }
    return season_dict


ps_3_year = [make_season(x) for x in datasets]


if __name__=="__main__":
    parse_numpy_dataset()

# def parse_effective_areas():
#     """Function to parse effective areas .txt into a format that flarestack
#     can use to build Signal/Background splines.
#     """
#     file_name = data_dir + "TabulatedAeff.txt"
#
#     data = []
#
#     with open(file_name, "r") as f:
#
#         csv_reader = csv.reader(f, delimiter=" ")
#
#         for i, row in enumerate(csv_reader):
#             if i > 0:
#                 row = [float(x) for x in row if x != ""]
#
#                 entry = tuple(row)
#
#                 data.append(entry)
#
#     data = np.array(data)
#
#
#     print(sorted(set(data[:,0] + data[:,1])))
#
#
# parse_effective_areas()

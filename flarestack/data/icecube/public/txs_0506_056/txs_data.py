import os
import numpy as np
import csv
import pickle
from flarestack.shared import public_dataset_dir
from flarestack.analyses.txs_0506_056.make_txs_catalogue import txs_catalogue

src_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

data_dir = src_dir + "raw_data/"
output_base_dir = public_dataset_dir + "txs_data/"
output_data_dir = output_base_dir + "events/"


for path in [output_data_dir]:
    try:
        os.makedirs(path)
    except OSError:
        pass


def data_path(season):
    return output_data_dir + season + ".npy"


data_dtype = np.dtype(
    [
        ("ra", float),
        ("dec", float),
        ("logE", float),
        ("sigma", float),
        ("time", float),
        ("sinDec", float),
    ]
)


datasets = ["IC40", "IC59", "IC79", "IC86a", "IC86b", "IC86c"]


def parse_numpy_dataset():
    """Function to parse the .txt file  of events into a numpy format
    readable by flarestack, which is the saved in the products/ subdirectory.
    """

    for dataset in datasets:
        data = []

        path = data_dir + "events_{0}.txt".format(dataset)

        with open(path, "r") as f:
            csv_reader = csv.reader(f, delimiter=" ")

            for i, row in enumerate(csv_reader):
                if i > 0:
                    row = [float(x) for x in row if x != ""]

                    entry = (
                        np.deg2rad(row[1]),
                        np.deg2rad(row[2]),
                        row[4],
                        np.deg2rad(row[3]),
                        row[0],
                        np.sin(np.deg2rad(row[2])),
                    )

                    data.append(entry)

        data = np.array(data, dtype=data_dtype)

        exp_path = data_path(dataset)

        with open(exp_path, "wb") as f:
            print("Saving converted numpy array to", exp_path)
            pickle.dump(data, f)


# Events are within 3 degrees of position of TXS 0505+056

dec = np.degrees(txs_catalogue["dec_rad"])
upper_sin_dec = np.sin(np.deg2rad(dec + 3.0))[0]
lower_sin_dec = np.sin(np.deg2rad(dec - 3.0))[0]

# Account for fact that not events are not distributed over RA range

ra_frac = np.deg2rad(6.0) / np.deg2rad(360.0)

sin_dec_bins = np.array([lower_sin_dec, upper_sin_dec])

print(sin_dec_bins.shape)
input("?")


def make_season(season_name):
    season_dict = {
        "Data Sample": "txs_data",
        "sinDec bins": sin_dec_bins,
        "MJD Time Key": "time",
        "Name": season_name,
        "exp_path": data_path(season_name),
        "mc_path": None,
        "grl_path": None,
    }
    return season_dict


txs_public_sample = [make_season(x) for x in datasets]


if __name__ == "__main__":
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

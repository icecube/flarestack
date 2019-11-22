import numpy as np
import os
import matplotlib.pyplot as plt
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
from flarestack.icecube_utils.dataset_loader import data_loader
from flarestack.shared import plots_dir, min_angular_err

ang_err_dir = plots_dir + "angular_error/"

for season in txs_sample_v1:
    sample_dir = ang_err_dir + season["Data Sample"] + "/"

    try:
        os.makedirs(sample_dir)
    except OSError:
        pass

    if isinstance(season["exp_path"], list):
        exp = np.concatenate(
            tuple([np.load(x) for x in season["exp_path"]]))
    else:
        exp = np.load(season["exp_path"])

    plt.figure()
    plt.hist(np.degrees(exp["sigma"]), bins=100)
    plt.yscale("log")
    plt.savefig(sample_dir + season["Name"] + "_RAW.pdf")
    plt.close()

    mask = exp["sigma"] < min_angular_err

    print()

    print(season["Name"], "(reading raw data)")

    print(np.sum(mask), "events have an angular error less than", end=' ')
    print(np.degrees(min_angular_err), "degrees")
    print("This is out of", len(mask), "total events")

    print()

    bins = np.linspace(-1, 1., 10)

    means = 0.5 * (bins[:-1] + bins[1:])
    fracs = []

    for i, lower in enumerate(bins[:-1]):
        upper = bins[i + 1]
        mask = np.sin(exp["dec"]) > lower
        mask *= np.sin(exp["dec"]) < upper

        band_data = exp["sigma"][mask]

        frac = np.sum(band_data < min_angular_err) / float(len(band_data))

        fracs.append(frac)

    plt.figure()
    plt.plot(means, fracs)
    plt.xlabel(r"$\sin(\delta)$")
    plt.ylabel(r"Fraction of events sub-0.2$^\circ$ error")
    plt.savefig(sample_dir + season["Name"] + "_declination.pdf")
    plt.close()

    t = exp[season["MJD Time Key"]]

    sidereal_day = 364./365.

    res = t % sidereal_day

    az_offset = 2.54199002505 + 2.907

    az = az_offset + 2 * np.pi * res - exp["ra"]

    while np.sum(az > 2 * np.pi) > 0:
        az[az > 2 * np.pi] -= 2 * np.pi

    while np.sum(az < 0) > 0:
        az[az < 0] += 2 * np.pi

    bins = np.linspace(0, 2*np.pi, 50)

    means = 0.5 * (bins[:-1] + bins[1:])
    fracs_north = []
    fracs_south = []

    for i, lower in enumerate(bins[:-1]):
        upper = bins[i + 1]
        mask = az > lower
        mask *= az < upper

        band_data = exp["sigma"][mask]

        north = band_data[exp["dec"][mask] > 0.]
        south = band_data[exp["dec"][mask] < 0.]

        frac_n = np.sum(north < min_angular_err) / float(len(north))

        fracs_north.append(frac_n)

        frac_s = np.sum(south < min_angular_err) / float(len(south))
        fracs_south.append(frac_s)

    plt.figure()
    plt.plot(means, fracs_north, label="Northern Sky")
    plt.plot(means, fracs_south, label="Southern Sky")
    plt.xlabel("azimuth")
    plt.ylabel(r"Fraction of events sub-0.2$^\circ$ error")
    plt.legend()
    plt.savefig(sample_dir + season["Name"] + "_azimuth.pdf")
    plt.close()

    bins = np.array(list(np.linspace(2.5, 5.0, 6)) + [7.0])

    means = 0.5 * (bins[:-1] + bins[1:])
    fracs_north = []
    fracs_south = []

    for i, lower in enumerate(bins[:-1]):
        upper = bins[i + 1]
        mask = exp["logE"] > lower
        mask *= exp["logE"] < upper

        band_data = exp["sigma"][mask]

        north = band_data[exp["dec"][mask] > 0.]
        south = band_data[exp["dec"][mask] < 0.]

        frac_n = np.sum(north < min_angular_err) / float(len(north))

        fracs_north.append(frac_n)

        frac_s = np.sum(south < min_angular_err) / float(len(south))
        fracs_south.append(frac_s)

        print("North", np.sum(north < min_angular_err), float(len(north)), \
            frac_n)

    plt.figure()
    plt.plot(means, fracs_north, label="Northern Sky")
    plt.plot(means, fracs_south, label="Southern Sky")
    plt.xlabel("log(E)")
    plt.ylabel(r"Fraction of events sub-0.2$^\circ$ error")
    plt.legend()
    plt.savefig(sample_dir + season["Name"] + "_energy.pdf")
    plt.close()

    exp = data_loader(season["exp_path"])

    mask = exp["sigma"] < min_angular_err

    plt.figure()
    plt.hist(np.degrees(exp["sigma"]), bins=100)
    plt.yscale("log")
    plt.savefig(sample_dir + season["Name"] + "_corrected.pdf")
    plt.close()

    print()

    print(season["Name"], "(read in with data_loader)")

    print(np.sum(mask), "events have an angular error less than", end=' ')
    print(np.degrees(min_angular_err), "degrees")
    print("This is out of", len(mask), "total events")

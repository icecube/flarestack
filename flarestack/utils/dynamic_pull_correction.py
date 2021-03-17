import numpy as np
import os
import pickle as Pickle
from flarestack.icecube_utils.dataset_loader import data_loader
from flarestack.core.energy_pdf import EnergyPDF
import matplotlib.pyplot as plt
from flarestack.shared import weighted_quantile, floor_pickle, pull_pickle
from flarestack.core.astro import angular_distance
from flarestack.utils.make_SoB_splines import get_gamma_support_points


def get_mc(floor_dict):
    return data_loader(floor_dict["season"]["mc_path"])


def get_pulls(mc):
    x = np.degrees(angular_distance(
        mc["ra"], mc["dec"], mc["trueRa"], mc["trueDec"]))
    y = np.degrees(mc["sigma"]) * 1.177
    return x / y


n_step = 50
min_loge_gap = 0.2

n_ebins = 5

# def create_quantile_floor(floor_dict):
#
#     path = floor_pickle(floor_dict)
#     print path
#
#     try:
#         floor_dim = int(floor_dict["floor_dim"])
#     except KeyError:
#         floor_dim = 0
#
#     if floor_dim == 0:
#         create_quantile_floor_0d(floor_dict)
#     else:
#         raise ValueError("Bad floor dimension{}".format(floor_dim))


def create_quantile_floor_0d(floor_dict):
    mc = get_mc(floor_dict)
    e_pdf = EnergyPDF.create(floor_dict["e_pdf_dict"])
    weights = e_pdf.weight_mc(mc)

    quantile_floor = weighted_quantile(
        mc["raw_sigma"], floor_dict["floor_quantile"], weights)

    save_path = floor_pickle(floor_dict)

    with open(save_path, "wb") as f:
        Pickle.dump(quantile_floor, f)

    print("Saved to", save_path)


def create_quantile_floor_0d_e(floor_dict):
    mc = get_mc(floor_dict)
    e_pdf = EnergyPDF.create(floor_dict["e_pdf_dict"])

    default, bounds, name = e_pdf.return_energy_parameters()

    if len(name) != 1:
        raise Exception("Trying to scan just one energy parameter, "
                        "but selected energy pdf gave the following parameters:"
                        " {} {} {}".format(name, default, bounds))

    x_range = np.linspace(bounds[0][0], bounds[0][1], n_step)
    y_range = []

    for x in x_range:
        weights = e_pdf.weight_mc(mc, x)
        quantile_floor = weighted_quantile(
            mc["raw_sigma"], floor_dict["floor_quantile"], weights)
        y_range.append(quantile_floor)

    y_range = np.array(y_range)

    save_path = floor_pickle(floor_dict)

    res = [x_range, np.log(y_range)]

    with open(save_path, "wb") as f:
        Pickle.dump(res, f)

    print("Saved to", save_path)

    plot_path = floor_pickle(floor_dict)[:-3] + "pdf"

    plt.figure()
    plt.plot(x_range, np.degrees(y_range))
    plt.savefig(plot_path)
    plt.close()


def create_quantile_floor_1d(floor_dict):

    mc = get_mc(floor_dict)
    e_pdf = EnergyPDF.create(floor_dict["e_pdf_dict"])
    weights = e_pdf.weight_mc(mc)

    bins = np.linspace(2., 6., 30)

    x_range = 0.5 * (bins[1:] + bins[:-1])
    y_range = []

    for j, lower in enumerate(bins[:-1]):
        upper = bins[j + 1]
        mask = np.logical_and(
            mc["logE"] >= lower,
            mc["logE"] < upper
        )
        quantile_floor = weighted_quantile(
            mc["raw_sigma"][mask], floor_dict["floor_quantile"], weights[mask])

        y_range.append(quantile_floor)

    x_range = np.array([0.] + list(x_range) + [10.])
    y_range = np.array([y_range[0]] + list(y_range) + [y_range[-1]])

    save_path = floor_pickle(floor_dict)
    res = [x_range, np.log(y_range)]

    with open(save_path, "wb") as f:
        Pickle.dump(res, f)
    print("Saved to", save_path)

    plot_path = floor_pickle(floor_dict)[:-3] + "pdf"

    plt.figure()
    plt.plot(x_range, np.degrees(y_range))
    plt.savefig(plot_path)
    plt.close()


def create_quantile_floor_1d_e(floor_dict):

    mc = get_mc(floor_dict)
    e_pdf = EnergyPDF.create(floor_dict["e_pdf_dict"])

    default, bounds, name = e_pdf.return_energy_parameters()

    if name != ["gamma"]:
        raise Exception("Trying to scan gamma parameter, "
                        "but selected energy pdf gave the following parameters:"
                        " {} {} {}".format(name, default, bounds))

    e_range = np.linspace(bounds[0][0], bounds[0][1], n_step)

    bins = np.linspace(2., 6., 20)

    z_range = []

    for j, lower in enumerate(bins[:-1]):
        upper = bins[j + 1]
        mask = np.logical_and(
            mc["logE"] >= lower,
            mc["logE"] < upper
        )

        cut_mc = mc[mask]

        vals = []

        for e in e_range:
            weights = e_pdf.weight_mc(cut_mc, e)

            quantile_floor = weighted_quantile(
                cut_mc["raw_sigma"], floor_dict["floor_quantile"], weights)

            vals.append(quantile_floor)

        z_range.append(vals)

    x_range = 0.5 * (bins[1:] + bins[:-1])
    x_range = np.array([0.] + list(x_range) + [10.])

    z_range = np.array([z_range[0]] + z_range + [z_range[-1]])

    save_path = floor_pickle(floor_dict)
    res = [x_range, e_range, np.log(z_range)]

    with open(save_path, "wb") as f:
        Pickle.dump(res, f)
    print("Saved to", save_path)

    from scipy.interpolate import RectBivariateSpline

    spline = RectBivariateSpline(
        x_range, e_range, np.log(np.degrees(z_range)),
        kx=1, ky=1, s=0)

    Z = []
    for x in x_range:
        Z.append(spline(x, e_range)[0])
    Z = np.array(Z)

    plot_path = floor_pickle(floor_dict)[:-3] + "pdf"

    ax = plt.subplot(111)
    X, Y = np.meshgrid(x_range, e_range)
    # cbar = ax.pcolor(X, Y, np.log(np.degrees(z_range.T)), cmap="viridis", )
    cbar = ax.pcolor(X, Y, Z.T, cmap="viridis", )
    plt.colorbar(cbar, label="Log(Angular Error Floor/deg)")
    plt.ylabel(name[0])
    plt.xlabel("Log(Energy proxy)")
    plt.savefig(plot_path)
    plt.close()


def create_pull_0d_e(pull_dict):
    mc = get_mc(pull_dict)
    pulls = get_pulls(mc)
    e_pdf = EnergyPDF.create(pull_dict["e_pdf_dict"])
    # gamma_precision = pull_dict.get('gamma_precision', 'flarestack')
    gamma_precision = pull_dict['gamma_precision']

    default, bounds, name = e_pdf.return_energy_parameters()

    if name != ["gamma"]:
        raise Exception("Trying to scan gamma parameter, "
                        "but selected energy pdf gave the following parameters:"
                        " {} {} {}".format(name, default, bounds))

    res_dict = dict()

    x_range = np.array(sorted(list(get_gamma_support_points(precision=gamma_precision))))

    y_range = []

    for x in x_range:
        weights = e_pdf.weight_mc(mc, x)
        median_pull = weighted_quantile(
            pulls, 0.5, weights)
        y_range.append(median_pull)
        res_dict[x] = np.log(median_pull)

    y_range = np.array(y_range)

    save_path = pull_pickle(pull_dict)
    plot_path = save_path[:-3] + "pdf"

    # print x_range, y_range

    plt.figure()
    plt.plot(x_range, y_range)
    plt.axhline(1.0, linestyle="--")
    plt.ylabel("Median Pull")
    plt.xlabel(name[0])
    plt.savefig(plot_path)
    plt.close()

    with open(save_path, "wb") as f:
        Pickle.dump(res_dict, f)

    print("Saved to", save_path)



def create_pull_1d(pull_dict):
    mc = get_mc(pull_dict)
    pulls = get_pulls(mc)
    e_pdf = EnergyPDF.create(pull_dict["e_pdf_dict"])
    weights = e_pdf.weight_mc(mc)

    bins = np.linspace(2., 6., n_ebins)

    x_range = 0.5 * (bins[1:] + bins[:-1])
    y_range = []

    for j, lower in enumerate(bins[:-1]):
        upper = bins[j + 1]
        mask = np.logical_and(
            mc["logE"] >= lower,
            mc["logE"] < upper
        )
        median_pull = weighted_quantile(
            pulls[mask], 0.5, weights[mask])
        y_range.append(median_pull)

    x_range = np.array([0.] + list(x_range) + [10.])
    y_range = np.array([y_range[0]] + list(y_range) + [y_range[-1]])

    save_path = pull_pickle(pull_dict)

    res = [x_range, np.log(y_range)]

    with open(save_path, "wb") as f:
        Pickle.dump(res, f)

    print("Saved to", save_path)

    plot_path = save_path[:-3] + "pdf"

    plt.figure()
    plt.plot(x_range, y_range)
    plt.axhline(1.0, linestyle="--")
    plt.ylabel("Median Pull")
    plt.xlabel("Log(Energy Proxy/GeV)")
    plt.savefig(plot_path)
    plt.close()


def create_pull_1d_e(floor_dict):

    mc = get_mc(floor_dict)
    pulls = get_pulls(mc)
    e_pdf = EnergyPDF.create(floor_dict["e_pdf_dict"])
    gamma_precision = floor_dict.get('gamma_precision', 'flarestack')

    default, bounds, name = e_pdf.return_energy_parameters()

    if name != ["gamma"]:
        raise Exception("Trying to scan gamma parameter, "
                        "but selected energy pdf gave the following parameters:"
                        " {} {} {}".format(name, default, bounds))

    e_range = np.array(sorted(list(get_gamma_support_points(precision=gamma_precision))))

    bins = np.linspace(2., 6., 5)
    x_range = 0.5 * (bins[1:] + bins[:-1])
    x_range = np.array([0.] + list(x_range) + [10.])

    res_dict = dict()

    z_range = []

    for e in sorted(e_range):
        weights = e_pdf.weight_mc(mc, e)

        vals = []

        for j, lower in enumerate(bins[:-1]):
            upper = bins[j + 1]
            mask = np.logical_and(
                mc["logE"] >= lower,
                mc["logE"] < upper
            )

            median_pull = weighted_quantile(
                pulls[mask], 0.5, weights[mask])

            vals.append(median_pull)

        vals = [vals[0]] + vals + [vals[-1]]

        z_range.append(vals)
        res_dict[e] = [x_range, np.log(vals)]

    z_range = np.array(z_range)

    # z_range = np.vstack((z_range.T[0], z_range.T)).T
    #
    # z_range = np.vstack((z_range.T, z_range.T[-1])).T

    save_path = pull_pickle(floor_dict)

    # res = [x_range, e_range, z_range]

    with open(save_path, "wb") as f:
        Pickle.dump(res_dict, f)
    print("Saved to", save_path)

    plot_path = save_path[:-3] + "pdf"

    ax = plt.subplot(111)
    X, Y = np.meshgrid(x_range, e_range)
    # cbar = ax.pcolor(X, Y, np.log(np.degrees(z_range.T)), cmap="viridis", )
    cbar = ax.pcolor(X, Y, np.log(z_range), cmap="seismic",
                     vmax=1.0, vmin=-1.0)
    plt.colorbar(cbar, label="Log(Median Pull)")
    plt.ylabel(name[0])
    plt.xlabel("Log(Energy proxy)")
    plt.savefig(plot_path)
    plt.close()

def create_pull_2d(pull_dict):
    mc = get_mc(pull_dict)
    pulls = get_pulls(mc)
    e_pdf = EnergyPDF.create(pull_dict["e_pdf_dict"])
    weights = e_pdf.weight_mc(mc)

    x_bins = np.linspace(2., 6., n_ebins)
    ymax = pull_dict["season"]["sinDec bins"][-1]
    ymin = pull_dict["season"]["sinDec bins"][0]
    # y_bins = np.linspace(ymin, ymax, 41)
    y_bins = pull_dict["season"]["sinDec bins"]
    x_range = 0.5 * (x_bins[1:] + x_bins[:-1])
    y_range = 0.5 * (y_bins[1:] + y_bins[:-1])

    z_range = np.ones([len(x_range), len(y_range)])

    for j, lower in enumerate(x_bins[:-1]):
        upper = x_bins[j + 1]
        mask = np.logical_and(
            mc["logE"] >= lower,
            mc["logE"] < upper
        )

        cut_mc = mc[mask]

        for k, lower_dec in enumerate(y_bins[:-1]):
            upper_dec = y_bins[k + 1]
            bin_mask = np.logical_and(
                cut_mc["sinDec"] >= lower_dec,
                cut_mc["sinDec"] < upper_dec
            )

            median_pull = weighted_quantile(
                pulls[mask][bin_mask], 0.5, weights[mask][bin_mask])

            z_range[j][k] = np.log(median_pull)

    # x_range = np.array([0.] + list(x_range) + [10.])
    # y_range = np.array([y_range[0]] + list(y_range) + [y_range[-1]])

    save_path = pull_pickle(pull_dict)

    res = [x_range, y_range, z_range]

    with open(save_path, "wb") as f:
        Pickle.dump(res, f)

    print("Saved to", save_path)

    plot_path = save_path[:-3] + "pdf"

    ax = plt.subplot(111)
    X, Y = np.meshgrid(x_bins, y_bins)
    cbar = ax.pcolor(X, Y, z_range.T, cmap="seismic",
                     vmax=1.0, vmin=-1.0)
    plt.colorbar(cbar, label="Log(Median Pull)")
    plt.ylabel(r"$\sin(\delta)$")
    plt.xlabel("Log(Energy proxy)")
    plt.savefig(plot_path)
    plt.close()

def create_pull_2d_e(pull_dict):
    save_path = pull_pickle(pull_dict)
    base_dir = save_path[:-3] + "/"

    try:
        os.makedirs(base_dir)
    except OSError:
        pass

    mc = get_mc(pull_dict)
    pulls = get_pulls(mc)
    e_pdf = EnergyPDF.create(pull_dict["e_pdf_dict"])
    gamma_precision = pull_dict.get('gamma_precision', 'flarestack')

    x_bins = np.linspace(2., 6., n_ebins)
    ymax = pull_dict["season"]["sinDec bins"][-1]
    ymin = pull_dict["season"]["sinDec bins"][0]
    y_bins = np.linspace(ymin, ymax, 11)

    x_range = 0.5 * (x_bins[1:] + x_bins[:-1])
    y_range = 0.5 * (y_bins[1:] + y_bins[:-1])

    res_dict = dict()

    e_range = np.array(sorted(list(get_gamma_support_points(precision=gamma_precision))))

    for e in e_range:

        z_range = np.ones([len(x_range), len(y_range)])

        weights = e_pdf.weight_mc(mc, e)

        for j, lower in enumerate(x_bins[:-1]):
            upper = x_bins[j + 1]
            mask = np.logical_and(
                mc["logE"] >= lower,
                mc["logE"] < upper
            )

            cut_mc = mc[mask]

            for k, lower_dec in enumerate(y_bins[:-1]):
                upper_dec = y_bins[k + 1]
                bin_mask = np.logical_and(
                    cut_mc["sinDec"] >= lower_dec,
                    cut_mc["sinDec"] < upper_dec
                )

                median_pull = weighted_quantile(
                    pulls[mask][bin_mask], 0.5, weights[mask][bin_mask])

                z_range[j][k] = np.log(median_pull)
        res_dict[e] = [x_range, y_range, z_range]

        plot_path = base_dir + str(e) + ".pdf"

        ax = plt.subplot(111)
        X, Y = np.meshgrid(x_bins, y_bins)
        cbar = ax.pcolor(X, Y, z_range.T, cmap="seismic",
                         vmax=1.0, vmin=-1.0)
        plt.colorbar(cbar, label="Log(Median Pull)")
        plt.ylabel(r"$\sin(\delta)$")
        plt.xlabel("Log(Energy proxy)")
        plt.savefig(plot_path)
        plt.close()


    save_path = pull_pickle(pull_dict)

    with open(save_path, "wb") as f:
        Pickle.dump(res_dict, f)

    print("Saved to", save_path)

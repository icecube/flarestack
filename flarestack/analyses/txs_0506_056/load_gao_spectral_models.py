import os
import numpy as np
import cPickle as Pickle
from astropy import units as u
from astropy import constants as const
from flarestack.shared import plot_output_dir, fs_scratch_dir
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

base_dir = "/afs/ifh.de/user/s/steinrob/scratch/gao_txs/"

output_name = "analyses/txs_0506_056/"

spline_dir = fs_scratch_dir + "splines/txs_0506_056/"

try:
    os.makedirs(spline_dir)
except OSError:
    pass


def spline_name(number):
    return spline_dir + "spline_" + str(number) + '.npy'


if __name__ == "__main__":
    xname = base_dir + "numu_x.dat"

    paths = [base_dir + x for x in os.listdir(base_dir) if x[0] != "."]
    paths.remove(xname)

    n_path = len(paths)

    convert_hz_ev = (u.Hz * const.h).to("GeV").value

    xvals = 10**np.loadtxt(xname) * convert_hz_ev

    savepath = plot_output_dir(output_name + "summary_plot.pdf")
    try:
        os.makedirs(os.path.dirname(savepath))
    except OSError:
        pass

    convert_ergs_GeV = (1. * u.erg).to("GeV").value

    plt.figure()
    for i in range(n_path):
        path = base_dir + "numu_y_" + str(i) + ".dat"
        yvals = 10**np.loadtxt(path) * (xvals**-2) * convert_ergs_GeV
        yvals[yvals < 10**-61] = 10**-61

        e_range = np.logspace(1, 7, 1e2)

        f = InterpolatedUnivariateSpline(xvals, np.log(yvals))

        path = spline_name(i)

        with open(path, "wb") as h:
            Pickle.dump(f, h)

        frac = float(i)/float(n_path)
        plt.plot(xvals, yvals*xvals**2, alpha=0.3, color=(1-frac, frac, 0))

    plt.ylim(10**-13, 10**-7)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Log(Energy/GeV)")
    plt.ylabel(r"E$^{2}\frac{dN}{dE}$ @ 1GeV [GeV cm$^{-2}$ s$^{-1}$]")
    plt.title("TXS 0506+056 Spectral Models")
    print "Saving to", savepath
    plt.savefig(savepath)
    plt.close()
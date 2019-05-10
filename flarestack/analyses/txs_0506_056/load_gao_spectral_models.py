from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import os
import numpy as np
import pickle as Pickle
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

    ax1 = plt.subplot(111)
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()

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
        ax1.plot(#np.log10(xvals), #/ convert_hz_ev),
                 xvals,
                 (yvals * xvals**2) / convert_ergs_GeV,
                 alpha=0.3, color=(1-frac, frac, 0))
        ax2.plot(xvals / convert_hz_ev,
                 (yvals * xvals**2) / convert_ergs_GeV, alpha=0.)
        ax3.plot(xvals, (yvals * xvals**2), alpha=0.)

    ax1.set_ylim(10**-14, 10**-9)
    ax3.set_ylim(10**-14*convert_ergs_GeV, 10**-9*convert_ergs_GeV)
    ax1.set_yscale("log")
    ax3.set_yscale("log")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    # ax2.set_xlim(10**24, 10**32)
    ax1.set_xlabel("Energy (GeV)")
    ax2.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel(r"E$^{2}\frac{dN}{dE}$ @ 1GeV [erg cm$^{-2}$ s$^{-1}$]")
    ax3.set_ylabel(r"E$^{2}\frac{dN}{dE}$ @ 1GeV [GeV cm$^{-2}$ s$^{-1}$]")
    ax1.grid(True)
    # plt.suptitle("TXS 0506+056 Spectral Models")
    plt.tight_layout()
    print("Saving to", savepath)
    plt.savefig(savepath)
    plt.close()

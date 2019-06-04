import cPickle as Pickle
import numpy as np
import matplotlib.pyplot as plt
from flarestack.shared import plot_output_dir
from scipy.interpolate import interp1d

# Upper limit from unblinding

data = {
    'fluence': [
        1.024657472792497, 1.0080136287356645, 1.0298688242795009,
        1.1901892139483494, 1.9355394666585095, 3.1582157516072162,
        5.885068012478378
    ],
    'energy': [
        7.620379276659313e+50, 7.496598981582775e+50, 7.659136106067608e+50,
        8.85143910243276e+50, 1.4394610133163642e+51, 2.348765563498741e+51,
        4.376726029411654e+51
    ],
    'flux': [
        7.270865724697601e-10, 2.3592549180040067e-09, 7.174904832216461e-09,
        2.2127098809628497e-08, 1.9146907812156704e-07, 1.270595350587128e-06,
        8.302460353325816e-06
    ],
    'x': [1.8, 1.9, 2.0, 2.1, 2.3, 2.5, 2.7]
}


f = interp1d(data["x"], np.log(data["energy"]))

x_range = np.linspace(data["x"][0], data["x"][-1], 10)

plt.figure()
plt.errorbar(x_range, np.exp(f(x_range)),
             yerr=.25 * np.exp(f(x_range)),
             uplims=True)
# plt.plot(x_range, np.exp(f(x_range)))
plt.ylabel(r"Per-flavour $E_{\nu}$ [erg]")
plt.xlabel(r"Spectral Index ($\gamma$)")
plt.yscale("log")
plt.title("AT2018cow neutrino emission limit (100GeV - 10PeV)")

plt.annotate("IceCube \n Preliminary ", (0.05, 0.75), alpha=0.5,
             fontsize=15,
             xycoords="axes fraction", multialignment="center")
plt.tight_layout()

savepath = plot_output_dir("analyses/tde/") + \
           "AT2018cow_upper_limit_fluence.pdf"

print "Saving to", savepath

plt.savefig(savepath)
plt.close()

"""This script contains the EnergyPDF classes, that are used for weighting
events based on a given energy PDF.

"""

import numexpr
import numpy as np
import cPickle as Pickle

gamma_range = [1., 4.]

default_emin = 100
default_emax = 10**7


class EnergyPDF:
    subclasses = {}

    def __init__(self, e_pdf_dict):
        if "E Min" in e_pdf_dict.keys():
            self.e_min = e_pdf_dict["E Min"]
            print "Minimum Energy is", self.e_min, "GeV."
            self.integral_e_min = self.e_min
        else:
            self.integral_e_min = default_emin

        if "E Max" in e_pdf_dict.keys():
            self.e_max = e_pdf_dict["E Max"]
            print "Maximum Energy is", self.e_max, "GeV."
            self.integral_e_max = self.e_max
        else:
            self.integral_e_max = default_emax


    @classmethod
    def register_subclass(cls, energy_pdf_name):
        """Adds a new subclass of EnergyPDF, with class name equal to
        "energy_pdf_name".
        """
        def decorator(subclass):
            cls.subclasses[energy_pdf_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, e_pdf_dict):
        e_pdf_name = e_pdf_dict["Name"]

        if e_pdf_name not in cls.subclasses:
            raise ValueError('Bad energy PDF name {}'.format(e_pdf_name))

        return cls.subclasses[e_pdf_name](e_pdf_dict)

    @staticmethod
    def f(energy):
        pass

    def integrate_over_E(self, f):
        """Uses Newton's method to integrate function f over the energy
        range. By default, uses 100GeV to 10PeV, unless otherwise specified.
        Uses 1000 logarithmically-spaced bins to calculate integral.

        :param f: Function to be integrated
        :return: Integral of function
        """

        nsteps = 1e3

        e_range = np.linspace(np.log(self.integral_e_min),
                              np.log(self.integral_e_max),
                              nsteps + 1)
        int_sum = 0.0

        for i, log_e in enumerate(e_range[:-1]):
            e0 = np.exp(log_e)
            e1 = np.exp(e_range[i + 1])
            int_sum += 0.5 * (e1 - e0) * (f(e0) + f(e1))

        return int_sum

    def flux_integral(self):
        """Integrates over energy PDF to give integrated flux (dN/dT)"""
        return self.integrate_over_E(self.f)

    def fluence_integral(self):
        """Performs an integral for fluence over a given energy range. This is
        gives the total energy per unit area per second that is radiated.
        """

        def g(energy):
            return energy * self.f(energy)

        return self.integrate_over_E(g)

    def return_energy_parameters(self):
        default = []
        bounds = []
        name = []
        return default, bounds, name


@EnergyPDF.register_subclass('Power Law')
class PowerLaw(EnergyPDF):
    """A Power Law energy PDF. Takes an argument of gamma in the dictionary
    for the init function, where gamma is the spectral index of the Power Law.
    """

    def __init__(self, e_pdf_dict={}):
        """Creates a PowerLaw object, which is an energy PDF based on a power
        law. The power law is generated from e_pdf_dict, which can specify a
        spectral index (Gamma), as well as an optional minimum energy (E Min)
        and a maximum energy (E Max)

        :param e_pdf_dict: Dictionary containing parameters
        """
        EnergyPDF.__init__(self, e_pdf_dict)
        if "Gamma" in e_pdf_dict.keys():
            self.gamma = float(e_pdf_dict["Gamma"])

    def weight_mc(self, mc, gamma=None):
        """Returns an array containing the weights for each MC event,
        given that the spectral index gamma has been chosen. Weights each
        event as (E/GeV)^-gamma, and multiplies this by the pre-existing MC
        oneweight value, to give the overall oneweight.

        :param mc: Monte Carlo
        :param gamma: Spectral Index (default is value in e_pdf_dict)
        :return: Weights Array
        """
        # Uses numexpr for faster processing
        ow = mc['ow']
        trueE = mc['trueE']
        if gamma is None:
            gamma = self.gamma
        weights = numexpr.evaluate('ow * trueE **(-gamma)')

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            mask = trueE < self.e_min
            weights[mask] = 0.

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            mask = trueE > self.e_max
            weights[mask] = 0.

        return weights

    def f(self, energy):
        val = energy ** -self.gamma

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            if energy < self.e_min:
                return 0.

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            if energy > self.e_max:
                return 0.

        return val

    def flux_integral(self):
        """Integrates over energy PDF to give integrated flux (dN/dT)"""

        # Integrate over flux to get dN/dt

        phi_power = 1 - self.gamma

        phi_integral = (1. / phi_power) * (
            (self.integral_e_max ** phi_power) - (
                self.integral_e_min ** phi_power)
        )

        return phi_integral

    def fluence_integral(self):
        """Performs an integral for fluence over a given energy range. This is
        gives the total energy per unit area per second that is radiated.
        """

        if self.gamma == 2:
            e_integral = np.log(self.integral_e_max /
                                self.integral_e_min)
        else:
            power = 2 - self.gamma

            # Get around astropy power rounding error (does not give
            # EXACTLY 2)

            e_integral = (
                (1. / power) * ((self.integral_e_max ** power) -
                                (self.integral_e_min ** power))
            )

        return e_integral

    def return_energy_parameters(self):
        default = [2.]
        bounds = [(gamma_range[0], gamma_range[1])]
        name = ["gamma"]
        return default, bounds, name

    def return_injected_parameters(self):
        return {"gamma": self.gamma}


@EnergyPDF.register_subclass('Spline')
class Spline(EnergyPDF):
    """A Power Law energy PDF. Takes an argument of gamma in the dictionary
    for the init function, where gamma is the spectral index of the Power Law.
    """

    def __init__(self, e_pdf_dict={}):
        """Creates a PowerLaw object, which is an energy PDF based on a power
        law. The power law is generated from e_pdf_dict, which can specify a
        spectral index (Gamma), as well as an optional minimum energy (E Min)
        and a maximum energy (E Max)

        :param e_pdf_dict: Dictionary containing parameters
        """
        EnergyPDF.__init__(self, e_pdf_dict)

        with open(e_pdf_dict["Spline Path"], "r") as g:
            f = Pickle.load(g)
            self.f = lambda x: np.exp(f(x))

    def weight_mc(self, mc):
        """Returns an array containing the weights for each MC event,
        given that the spectral index gamma has been chosen. Weights each
        event using the energy spline, and multiplies this by the
        pre-existing MC oneweight value, to give the overall oneweight.

        :param mc: Monte Carlo
        :return: Weights Array
        """
        # Uses numexpr for faster processing
        weights = mc['ow'] * self.f(mc['trueE'])

        # If there is a minimum energy, gives a weight of 0 to events below
        if hasattr(self, "e_min"):
            mask = mc['trueE'] < self.e_min
            weights[mask] = 0.

        # If there is a maximum energy, gives a weight of 0 to events above
        if hasattr(self, "e_max"):
            mask = mc['trueE'] > self.e_max
            weights[mask] = 0.

        return weights


# if __name__ == "__main__":
#
#     from flarestack.shared import plots_dir, fs_scratch_dir
#     from astropy.modeling.powerlaws import LogParabola1D
#     import matplotlib.pyplot as plt
#     from scipy.interpolate import InterpolatedUnivariateSpline
#
#     g = EnergyPDF.create(
#         {
#             "Name": "Power Law",
#             "Gamma": 2
#         }
#     )
#
#     e_range = np.logspace(0, 7, 1e3)
#
#     f = InterpolatedUnivariateSpline(e_range, np.log(g.f(e_range)))
#
#     path = fs_scratch_dir + "tester_spline.npy"
#
#     print path
#
#     with open(path, "wb") as h:
#         Pickle.dump(f, h)
#
#     e_pdf_dict = {
#         "Name": "Spline",
#         "Spline Path": path,
#     }
#
#     energy_pdf = EnergyPDF.create(e_pdf_dict)
#
#     plt.figure()
#     plt.plot(e_range, energy_pdf.f(e_range))
#     plt.plot(e_range, g.f(e_range))
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.savefig(plots_dir + "spline.pdf")
#     plt.close()
#
#     print "Directly from Power Law"
#
#     print g.fluence_integral()
#     print g.flux_integral()
#
#     print "Via Spline"
#
#     print energy_pdf.fluence_integral()
#     print energy_pdf.flux_integral()





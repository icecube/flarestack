"""This script contains the EnergyPDF classes, that are used for weighting
events based on a given energy PDF.

"""

import numexpr
import numpy as np


class EnergyPDF:
    subclasses = {}

    def __init__(self):
        pass

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


@EnergyPDF.register_subclass('Power Law')
class PowerLaw(EnergyPDF):
    """A Power Law energy PDF. Takes an argument of gamma in the dictionary
    for the init function, where gamma is the spectral index of the Power Law.
    """

    def __init__(self, e_pdf_dict=dict()):
        """Creates a PowerLaw object, which is an energy PDF based on a power
        law. The power law is generated from e_pdf_dict, which can specify a
        spectral index (Gamma), as well as an optional minimum energy (E Min)
        and a maximum energy (E Max)

        :param e_pdf_dict: Dictionary containing parameters
        """
        if "Gamma" in e_pdf_dict.keys():
            self.gamma = float(e_pdf_dict["Gamma"])

        if "E Min" in e_pdf_dict.keys():
            self.e_min = e_pdf_dict["E Min"]
            print "Minimum Energy is", self.e_min, "GeV."

        if "E Max" in e_pdf_dict.keys():
            self.e_max = e_pdf_dict["E Max"]
            print "Maximum Energy is", self.e_max, "GeV."

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




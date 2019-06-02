"""Utility function to give approximated background atmospheric neutrino
spectrum. To first order, this is a power law with index $E^{-3.7}$.
"""

def approximated_atmo_spectrum(energy):
    """Gives an approximated atmospheric neutrino spectrum.
    Can be used for comparing expected true energy distribution to recorded
    energy proxy distributions. It is normalised such that the weight for an
    energy of 1 is equal to 1. (It is agnostic to energy units)

    :param energy: True neutrino energy (in some consistent unit)
    :return: Spectrum weight for that energy
    """
    return energy ** -3.7

from flarestack.shared import catalogue_dir
import os
tde_dir = os.path.abspath(os.path.dirname(__file__))

individual_tdes = [
    # "Swift J1644+57",
    # "Swift J2058+05",
    # "ASASSN-14li",
    # "XMMSL1 J0740-85",
    "AT2018cow"
]


def individual_tde_cat(name):
    """Maps the name of a given TDE to the path of a source catalogue which
    contains only that TDE.

    :param name: Name of TDE
    :return: Path to catalogue
    """
    return catalogue_dir + "TDEs/individual_TDEs/" + name + "_catalogue.npy"


tde_catalogues = [
    "jetted",
    "gold",
    "silver",
    "obscured"
]


def tde_catalogue_name(name):
    """Maps the name of a given TDE catalogue to the path of a source
    catalogue which contains those TDEs in one combined catalogue.

    :param name: Name of TDE catalogue
    :return: Path to catalogue
    """

    return tde_dir + "/tde_catalogues/TDE_" + name + "_catalogue.npy"

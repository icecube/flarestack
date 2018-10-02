from flarestack.shared import catalogue_dir

individual_tdes = [
    "Swift J1644+57",
    "Swift J2058+05",
    "ASASSN-14li",
    "XMMSL1 J0740-85"
    # "ASASSN-15lh",
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
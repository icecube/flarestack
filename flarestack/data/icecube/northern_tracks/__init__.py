import numpy as np
from flarestack.data.icecube.ic_season import IceCubeSeason
from numpy.lib.recfunctions import rename_fields


diffuse_binning = {
    "IC59": (
        np.unique(np.concatenate([
            np.linspace(0.0, 0.2, 14 + 1),
            np.linspace(0.2, 0.9, 24 + 1),
            np.linspace(0.9, 1.0, 4 + 1)
        ])),
        np.linspace(-1.5, 2., 40 + 1)
    ),
    "IC79": (
        np.unique(np.concatenate([
            np.linspace(np.sin(-np.radians(5)), 0.2, 24 + 1),
            np.linspace(0.2, 0.9, 24 + 1),
            np.linspace(0.9, 1.0, 4 + 1)
        ])),
        np.linspace(2., 7., 40 + 1) + 0.2
    ),
    "IC86": (
        np.unique(np.concatenate([
            np.linspace(np.sin(-np.radians(5)), 0.2, 24 + 1),
            np.linspace(0.2, 0.9, 24 + 1),
            np.linspace(0.9, 1.0, 4 + 1)
        ])),
        np.linspace(2., 7., 40 + 1)
    )
}


def get_diffuse_binning(season):
    if "IC86" in season:
        season = "IC86"
    return diffuse_binning[season]


class NTSeason(IceCubeSeason):

    def get_background_model(self):
        mc = self.load_data(self.mc_path, cut_fields=False)
        mc = rename_fields(mc, {"conv": "weight"})
        return mc

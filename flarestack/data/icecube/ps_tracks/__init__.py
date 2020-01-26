import numpy as np
from flarestack.data.icecube.ic_season import IceCubeDataset, IceCubeSeason

# Sets transition between North/South

boundary = np.sin(np.radians(-5.))

ps_binning = {
    # "IC40": (
    #     np.unique(np.concatenate([
    #         np.linspace(-1., -0.25, 10 + 1),
    #         np.linspace(-0.25, 0.0, 10 + 1),
    #         np.linspace(0.0, 1., 10 + 1),])),
    #     np.arange(2., 9. + 0.01, 0.125)
    # ),
    "IC40": (
        np.unique(np.concatenate([
            np.linspace(-1., -0.25, 5 + 1),
            np.linspace(-0.25, 0.0, 5 + 1),
            np.linspace(0.0, 1., 5 + 1), ])),
        np.arange(2., 9. + 0.01, 0.25)
    ),
    "IC59": (
        np.unique(np.concatenate([
            np.linspace(-1., -0.95, 2 + 1),
            np.linspace(-0.95, -0.25, 25 + 1),
            np.linspace(-0.25, 0.05, 15 + 1),
            np.linspace(0.05, 1., 10 + 1),
        ])),
        np.arange(2., 9.5 + 0.01, 0.125)
    ),
    "IC79": (
        np.linspace(-1., 1., 50),
        np.arange(2., 9. + 0.01, 0.125)
    ),
    "IC86": (
        np.unique(np.concatenate([
                np.linspace(-1., -0.2, 10 + 1),
                np.linspace(-0.2, boundary, 4 + 1),
                np.linspace(boundary, 0.2, 5 + 1),
                np.linspace(0.2, 1., 10)
            ])),
        np.arange(1., 10. + 0.01, 0.125)
    )
}


def get_ps_binning(season):
    if "IC86" in season:
        season = "IC86"
    return ps_binning[season]


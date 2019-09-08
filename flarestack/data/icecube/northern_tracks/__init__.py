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

    def simulate_background(self):
        base = self.get_background_model()

        n_exp = np.sum(base["weight"])

        # Simulates poisson noise around the expectation value n_inj.
        n_bkg = np.random.poisson(n_exp)

        # Creates a normalised array of OneWeights
        p_select = base['weight'] / n_exp

        # Creates an array with n_signal entries.
        # Each entry is a random integer between 0 and no. of sources.
        # The probability for each integer is equal to the OneWeight of
        # the corresponding source_path.
        ind = np.random.choice(len(base['ow']), size=n_bkg, p=p_select)

        # Selects the sources corresponding to the random integer array
        sim_bkg = base[ind]
        sim_bkg = sim_bkg[list(self.get_background_dtype().names)]
        return sim_bkg

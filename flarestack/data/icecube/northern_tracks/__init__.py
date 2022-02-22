import numpy as np
from flarestack.data.icecube.ic_season import IceCubeSeason
from numpy.lib.recfunctions import rename_fields


diffuse_binning = {
    "IC59": (
        np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 0.2, 14 + 1),
                    np.linspace(0.2, 0.9, 24 + 1),
                    np.linspace(0.9, 1.0, 4 + 1),
                ]
            )
        ),
        np.linspace(-1.5, 2.0, 40 + 1),
    ),
    "IC79": (
        np.unique(
            np.concatenate(
                [
                    np.linspace(np.sin(-np.radians(5)), 0.2, 24 + 1),
                    np.linspace(0.2, 0.9, 24 + 1),
                    np.linspace(0.9, 1.0, 4 + 1),
                ]
            )
        ),
        np.linspace(2.0, 7.0, 40 + 1) + 0.2,
    ),
    "IC86": (
        np.unique(
            np.concatenate(
                [
                    np.linspace(np.sin(-np.radians(5)), 0.2, 24 + 1),
                    np.linspace(0.2, 0.9, 24 + 1),
                    np.linspace(0.9, 1.0, 4 + 1),
                ]
            )
        ),
        np.linspace(2.0, 7.0, 40 + 1),
    ),
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
        if isinstance(self.loaded_background_model, type(None)):
            self.load_background_model()

        # base = self.get_background_model()

        n_exp = np.sum(self.loaded_background_model["weight"])
        # n_exp = np.sum(base["weight"])

        # Simulates poisson noise around the expectation value n_inj.
        n_bkg = np.random.poisson(n_exp)

        # Creates a normalised array of OneWeights
        p_select = self.loaded_background_model["weight"] / n_exp
        # p_select = base['weight'] / n_exp

        # Creates an array with n_signal entries.
        # Each entry is a random integer between 0 and no. of sources.
        # The probability for each integer is equal to the OneWeight of
        # the corresponding source_path.
        ind = np.random.choice(
            len(self.loaded_background_model["ow"]), size=n_bkg, p=p_select
        )
        # ind = np.random.choice(len(base['ow']), size=n_bkg, p=p_select)

        # Selects the sources corresponding to the random integer array
        sim_bkg = self.loaded_background_model[ind]
        # sim_bkg = base[ind]
        sim_bkg = sim_bkg[list(self.get_background_dtype().names)]
        return sim_bkg


class NTSeasonNewStyle(NTSeason):
    def get_background_model(self):
        # in version 3 of the dataset the weights are given as rates in Hz
        # instead of total events in the associated livetime
        mc = super(NTSeasonNewStyle, self).get_background_model()
        livetime = self.get_time_pdf().get_livetime()
        mc["astro"] *= livetime * 86400.0
        mc["weight"] *= livetime * 86400.0
        mc["prompt"] *= livetime * 86400.0
        return mc

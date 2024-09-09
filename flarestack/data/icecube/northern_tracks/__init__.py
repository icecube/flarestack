import numpy as np
from astropy.table import Table
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
    def get_background_model(self) -> Table:
        """Loads Monte Carlo dataset from file according to object path set in object properties.

        Returns:
            dict: Monte Carlo data set.
        """
        mc = self.load_data(self.mc_path, cut_fields=False)
        # According to NT specifications (README):
        #  "conv" gives the weight for conventional atmospheric neutrinos
        #  flarestack renames it to "weight"
        mc.rename_column("conv", "weight")
        return mc

    def simulate_background(self):
        rng = np.random

        if self.loaded_background_model is None:
            raise RuntimeError(
                "Monte Carlo background is not loaded. Call `load_background_model` before `simulate_background`."
            )

        n_mc = len(self.loaded_background_model["weight"])

        # Total number of events in the MC sample, weighted according to background.
        n_exp = np.sum(self.loaded_background_model["weight"])

        # Creates a normalised array of atmospheric weights.
        p_select = self.loaded_background_model["weight"] / n_exp

        # Simulates poisson noise around the expectation value n_exp.
        n_bkg = rng.poisson(n_exp)

        # Choose n_bkg from n_mc events according to background weight.
        ind = rng.choice(n_mc, size=n_bkg, p=p_select)
        sim_bkg = self.loaded_background_model[ind]

        time_pdf = self.get_time_pdf()

        # Simulates random times
        sim_bkg["time"] = time_pdf.simulate_times(source=None, n_s=n_bkg)

        # Check that the time pdf evaluates to 1 for all the simulated times.
        pdf_sum = np.sum(time_pdf.season_f(sim_bkg["time"]))
        if pdf_sum < n_bkg:
            raise RuntimeError(
                f"The time PDF does not evaluate to 1 for all generated event times.\n \
                The sum of the PDF values over {n_bkg} events is {pdf_sum}.\n \
                This means the sampling of background times is not reliable and must be fixed."
            )

        # Reduce the data to the relevant fields for analysis.
        analysis_keys = list(self.get_background_dtype().names)
        return sim_bkg[analysis_keys]


class NTSeasonNewStyle(NTSeason):
    def get_background_model(self):
        # in version >=3 of the dataset the weights are given as rates in Hz
        # instead of total events in the associated livetime
        # we deal with this by overwriting the MC set
        # possibly not the best course of action
        mc = super(NTSeasonNewStyle, self).get_background_model()
        livetime = self.get_time_pdf().get_livetime()
        for weight in ("astro", "weight", "prompt"):
            mc[weight] = mc[weight] * livetime * 86400.0
        return mc

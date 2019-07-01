from flarestack.data.public import ps_3_year
from flarestack.data.simulate import SimSeason, SimDataset
import numpy as np


class SimCubeSeason(SimSeason):

    def __init__(self, season_name, sample_name, pseudo_mc_path,
                 effective_area_f, bkg_time_pdf_dict, bkg_flux_norm,
                 bkg_e_pdf_dict, energy_proxy_map, sin_dec_bins,
                 log_e_bins, **kwargs):

        self.log_e_bins = log_e_bins
        self.sin_dec_bins = sin_dec_bins

        SimSeason.__init__(
            self, season_name, sample_name, pseudo_mc_path,
            effective_area_f, bkg_time_pdf_dict, bkg_flux_norm,
            bkg_e_pdf_dict, energy_proxy_map, **kwargs
        )

    def generate_sim_data(self):

        for i, lower_sin_dec in enumerate(self.sin_dec_bins[:-1]):
            upper_sin_dec = self.sin_dec_bins[i + 1]
            new_events = self.simulate_dec_range(lower_sin_dec, upper_sin_dec)

        return None

    def simulate_dec_range(self, lower_sin_dec, upper_sin_dec):
        mean_sin_dec = 0.5 * (lower_sin_dec + upper_sin_dec)
        solid_angle = 2 * np.pi * (upper_sin_dec - lower_sin_dec)
        print(lower_sin_dec, upper_sin_dec, solid_angle)



    # def simulate(self, mjd_start, mjd_end):
    #     time = (mjd_end - mjd_start) * 60 * 60 * 24
    #
    #     print("{0} seconds to simulate (MJD {1} to MJD {2})".format(
    #         time, mjd_start, mjd_end
    #     ))
    #
    #     flux = self.bkg_flux_norm * time
    #
    #     print(self.log_e_bins, self.sin_dec_bins)
    #
    #     self.effective_area_f = self.load_effective_area()
    #
    #     print(self.effective_area_f(0.0, 0.0))
    #
    #     print(flux)


e_pdf_dict = {
    "name": "PowerLaw",
    "gamma": 3.7
}

simcube_dataset = SimDataset()

for (name, season) in ps_3_year.get_seasons().items():

    def ideal_energy_proxy(e):
        return np.log10(e)

    def wrapper_f(mjd_start, mjd_end, bkg_flux_norm, bkg_e_pdf_dict,
                  energy_proxy_map=None, sim_name=None, **kwargs):

        if np.logical_and(energy_proxy_map is None, sim_name is None):
            energy_proxy_map = ideal_energy_proxy
            sim_name = "default"

        if np.logical_and(energy_proxy_map != ideal_energy_proxy,
                          sim_name is None):
            raise ValueError("Non-default energy proxy mapping was used, "
                             "but no unique sim_name was provided. Please "
                             "provide a unique 'sim_name' to dewcrive this "
                             "simulation.")

        sim_season = SimCubeSeason(
            season_name=name,
            sample_name="SimCube_{0}".format(sim_name),
            pseudo_mc_path=season.pseudo_mc_path,
            effective_area_f=season.load_effective_area(),
            mjd_start=mjd_start,
            mjd_end=mjd_end,
            bkg_flux_norm=bkg_flux_norm,
            bkg_e_pdf_dict=bkg_e_pdf_dict,
            energy_proxy_map=energy_proxy_map,
            sin_dec_bins=season.sin_dec_bins,
            log_e_bins=season.log_e_bins
        )
        return sim_season

    simcube_dataset.add_sim_season(name, wrapper_f)

simcube_season = simcube_dataset.set_sim_params(
    name="IC86-2012",
    mjd_start=55000.,
    mjd_end=55100.,
    bkg_flux_norm=1.,
    bkg_e_pdf_dict=e_pdf_dict
)

print(simcube_dataset.get_seasons())

# nicecube_10year = SimCubeSeason(0, 100, 1., e_pdf_dict)
#

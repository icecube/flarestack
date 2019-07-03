from flarestack.data.public import ps_3_year
from flarestack.data.simulate import SimSeason, SimDataset
import numpy as np
import random
from scipy.interpolate import interp1d


class SimCubeSeason(SimSeason):

    def __init__(self, season_name, sample_name, pseudo_mc_path,
                 event_dtype, effective_area_f, load_angular_resolution,
                 bkg_time_pdf_dict, bkg_flux_norm, bkg_e_pdf_dict,
                 energy_proxy_map, sin_dec_bins, log_e_bins, **kwargs):

        self.log_e_bins = log_e_bins
        self.sin_dec_bins = sin_dec_bins

        SimSeason.__init__(
            self, season_name, sample_name, pseudo_mc_path,
            event_dtype, effective_area_f,
            load_angular_resolution, bkg_time_pdf_dict, bkg_flux_norm,
            bkg_e_pdf_dict, energy_proxy_map, **kwargs
        )

    def generate_sim_data(self, fluence):
        print("Simulating events:")
        sim_events = np.empty((0,),
                              dtype=self.event_dtype)

        for i, lower_sin_dec in enumerate(self.sin_dec_bins[:-1]):
            upper_sin_dec = self.sin_dec_bins[i + 1]
            new_events = self.simulate_dec_range(
                fluence,lower_sin_dec, upper_sin_dec)

            print("Simulated {0} events between sin(dec)={1} and "
                  "sin(dec)={2}".format(
                len(new_events), lower_sin_dec, upper_sin_dec))

            # Joins the new events to the signal events
            sim_events = np.concatenate((sim_events, new_events))

        sim_events = np.array(sim_events)

        print("Simulated {0} events in total".format(len(sim_events)))

        return sim_events

    def simulate_dec_range(self, fluence, lower_sin_dec, upper_sin_dec):
        mean_sin_dec = 0.5 * (lower_sin_dec + upper_sin_dec)
        solid_angle = 2 * np.pi * (upper_sin_dec - lower_sin_dec)
        sim_fluence = fluence * solid_angle # GeV^-1 cm^-2

        def source_eff_area(e):
            return self.effective_area_f(
                np.log10(e), mean_sin_dec) * self.bkg_energy_pdf.f(e)

        int_eff_a = self.bkg_energy_pdf.integrate_over_E(source_eff_area)
        print(int_eff_a)

        # Effective areas are given in m2, but flux is in per cm2

        int_eff_a *= 10**4

        n_exp = sim_fluence * int_eff_a
        n_sim = np.random.poisson(n_exp)

        new_events = np.empty((n_sim,), dtype=self.event_dtype)

        new_events["ra"] = [random.random() * 2 * np.pi for _ in range(n_sim)]

        new_events["sinDec"] = [
            lower_sin_dec + (random.random() * (upper_sin_dec - lower_sin_dec))
            for _ in range(n_sim)
        ]
        new_events["dec"] = np.arcsin(new_events["sinDec"])
        new_events["time"] = self.time_pdf.simulate_times([], n_sim)

        fluence_ints, log_e_range = \
            self.bkg_energy_pdf.piecewise_integrate_over_energy(
            source_eff_area
        )
        fluence_ints = fluence_ints

        fluence_ints = np.array(fluence_ints)
        fluence_ints /= np.sum(fluence_ints)

        fluence_cumulative = [np.sum(fluence_ints[:i])
                              for i, _ in enumerate(fluence_ints)]

        fluence_cumulative = [0.] + fluence_cumulative + [1.]

        log_e_range = list(log_e_range) + [log_e_range[-1]]

        sim_true_e = interp1d(fluence_cumulative, log_e_range)

        true_e_vals = np.array(
            [10**sim_true_e(random.random()) for _ in range(n_sim)])

        new_events["logE"] = self.energy_proxy_map(true_e_vals)

        new_events["sigma"] = self.angular_res_f(new_events["logE"]).copy()
        new_events["raw_sigma"] = new_events["sigma"].copy()

        return new_events


e_pdf_dict = {
    "energy_pdf_name": "PowerLaw",
    "gamma": 1.7,
    # "e_min_gev": 1000,
}

simcube_dataset = SimDataset()

for (name, season) in ps_3_year.get_seasons().items():

    def ideal_energy_proxy(e):
        return np.log10(e)

    def wrapper_f(bkg_time_pdf_dict, bkg_flux_norm, bkg_e_pdf_dict,
                  energy_proxy_map=None, sim_name=None, **kwargs):

        if np.logical_and(energy_proxy_map is None, sim_name is None):
            energy_proxy_map = ideal_energy_proxy
            sim_name = "default"

        if np.logical_and(energy_proxy_map != ideal_energy_proxy,
                          sim_name is None):
            raise ValueError("Non-default energy proxy mapping was used, "
                             "but no unique sim_name was provided. Please "
                             "provide a unique 'sim_name' to describe this "
                             "simulation.")

        sim_season = SimCubeSeason(
            season_name=name,
            sample_name="SimCube_{0}".format(sim_name),
            pseudo_mc_path=season.pseudo_mc_path,
            effective_area_f=season.load_effective_area(),
            event_dtype=season.get_background_dtype(),
            load_angular_resolution=season.load_angular_resolution,
            bkg_time_pdf_dict=bkg_time_pdf_dict,
            bkg_flux_norm=bkg_flux_norm,
            bkg_e_pdf_dict=bkg_e_pdf_dict,
            energy_proxy_map=energy_proxy_map,
            sin_dec_bins=season.sin_dec_bins,
            log_e_bins=season.log_e_bins
        )
        return sim_season

    simcube_dataset.add_sim_season(name, wrapper_f)

bkg_time_pdf_dict = {
    "time_pdf_name": "fixed_end_box",
    "start_time_mjd": 50000,
    "end_time_mjd": 50020
}

simcube_season = simcube_dataset.set_sim_params(
    name="IC86-2012",
    bkg_time_pdf_dict=bkg_time_pdf_dict,
    bkg_flux_norm=1.,
    bkg_e_pdf_dict=e_pdf_dict
)

print(simcube_dataset.get_seasons())

# nicecube_10year = SimCubeSeason(0, 100, 1., e_pdf_dict)
#

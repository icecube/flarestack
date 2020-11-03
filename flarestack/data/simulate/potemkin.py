import numpy as np
import random
from scipy.interpolate import interp1d
import logging
from flarestack.data.public import icecube_ps_3_year
from flarestack.core.energy_pdf import EnergyPDF
from flarestack.data.simulate import SimSeason, SimDataset
from flarestack.shared import flux_to_k
from flarestack.data.public.icecube import PublicICSeason

logger = logging.getLogger(__name__)

class BackgroundFluxModel:


    def __init__(self, flux_norm, bkg_time_pdf_dict):
        self.flux_norm = flux_to_k(flux_norm)
        self.bkg_time_pdf_dict = bkg_time_pdf_dict

    def get_norm(self):
        return self.flux_norm

    def flux_model_f(self, e, sindec):
        return NotImplementedError

    def flux_range(self):
        return NotImplementedError

    def unique_name(self):
        return self.flux_norm

    def build_time_pdf_dict(self):
        return self.bkg_time_pdf_dict

class IdealBackgroundFluxModel(BackgroundFluxModel):

    def __init__(self, flux_norm, bkg_time_pdf_dict):
        BackgroundFluxModel.__init__(self, flux_norm, bkg_time_pdf_dict)
        self.atmo = self.atmo_flux()

    @staticmethod
    def atmo_flux():
        bkg_e_pdf_dict = {
            "energy_pdf_name": "power_law",
            "gamma": 3.7,
            "e_min_gev": 100.,
            "e_max_gev": 10.**7
        }
        return EnergyPDF.create(bkg_e_pdf_dict)

    def flux_model_f(self, e, sindec):
        return self.atmo.f(e)

    def flux_range(self):
        return self.atmo.e_min, self.atmo.e_max


class PotemkinSeason(SimSeason):

    def __init__(self, season_name, sample_name, pseudo_mc_path,
                 event_dtype, load_effective_area, load_angular_resolution,
                 bkg_flux_model,
                 energy_proxy_map, sin_dec_bins, log_e_bins, a_eff_path, **kwargs):

        self.log_e_bins = log_e_bins
        self.sin_dec_bins = sin_dec_bins
        self.a_eff_path = a_eff_path

        SimSeason.__init__(
            self, season_name, sample_name, pseudo_mc_path,
            event_dtype, load_effective_area,
            load_angular_resolution, bkg_flux_model,
            energy_proxy_map, **kwargs
        )

    def generate_sim_data(self, fluence):
        logger.info("Simulating events:")
        sim_events = np.empty((0,),
                              dtype=self.event_dtype)

        for i, lower_sin_dec in enumerate(self.sin_dec_bins[:-1]):
            upper_sin_dec = self.sin_dec_bins[i + 1]
            new_events = self.simulate_dec_range(
                fluence,lower_sin_dec, upper_sin_dec)

            logger.info("Simulated {0} events between sin(dec)={1} and "
                  "sin(dec)={2}".format(
                len(new_events), lower_sin_dec, upper_sin_dec))

            # Joins the new events to the signal events
            sim_events = np.concatenate((sim_events, new_events))

        sim_events = np.array(sim_events)

        logger.info("Simulated {0} events in total".format(len(sim_events)))

        return sim_events

    def simulate_dec_range(self, fluence, lower_sin_dec, upper_sin_dec):
        mean_sin_dec = 0.5 * (lower_sin_dec + upper_sin_dec)

        solid_angle = 2 * np.pi * (upper_sin_dec - lower_sin_dec)
        sim_fluence = fluence * solid_angle # GeV^-1 cm^-2

        effective_area_f = self.load_effective_area()

        def source_eff_area(e):
            return effective_area_f(
                np.log10(e), mean_sin_dec) * self.bkg_flux_model.flux_model_f(e, mean_sin_dec)

        lower, upper = self.bkg_flux_model.flux_range()

        logger.debug(f"Simulating between {lower:.2g} GeV and {upper:.2g} GeV")

        int_eff_a = EnergyPDF.integrate(source_eff_area, lower, upper)

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
        new_events["time"] = self.get_time_pdf().simulate_times([], n_sim)

        fluence_ints, log_e_range = \
            EnergyPDF.piecewise_integrate(
            source_eff_area, lower, upper
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
            [10.**sim_true_e(random.random()) for _ in range(n_sim)])

        new_events["logE"] = self.energy_proxy_map(true_e_vals)

        angular_res_f = self.load_angular_resolution()

        new_events["sigma"] = angular_res_f(new_events["logE"]).copy()
        new_events["raw_sigma"] = new_events["sigma"].copy()

        return new_events

    def make_season(self):
        return PublicICSeason(
            season_name=self.season_name,
            sample_name=self.sample_name,
            exp_path=self.exp_path,
            pseudo_mc_path=self.pseudo_mc_path,
            sin_dec_bins=self.sin_dec_bins,
            log_e_bins=self.log_e_bins,
            a_eff_path=self.a_eff_path,
        )

potemkin_dataset = SimDataset()

for (name, season) in icecube_ps_3_year.get_seasons().items():

    def ideal_energy_proxy(e):
        return np.log10(e)

    def wrapper_f(bkg_flux_model, energy_proxy_map=None, sim_name=None, **kwargs):

        if np.logical_and(energy_proxy_map is None, sim_name is None):
            energy_proxy_map = ideal_energy_proxy
            sim_name = "default"

        if np.logical_and(energy_proxy_map != ideal_energy_proxy,
                          sim_name is None):
            raise ValueError("Non-default energy proxy mapping was used, "
                             "but no unique sim_name was provided. Please "
                             "provide a unique 'sim_name' to describe this "
                             "simulation.")

        sin_dec_bins = season.sin_dec_bins[season.sin_dec_bins > 0.]

        sim_season = PotemkinSeason(
            season_name=name,
            sample_name="SimCube_{0}".format(sim_name),
            pseudo_mc_path=season.pseudo_mc_path,
            load_effective_area=season.load_effective_area,
            event_dtype=season.get_background_dtype(),
            load_angular_resolution=season.load_angular_resolution,
            bkg_flux_model=bkg_flux_model,
            energy_proxy_map=energy_proxy_map,
            sin_dec_bins=sin_dec_bins,
            log_e_bins=season.log_e_bins,
            a_eff_path=season.a_eff_path,
            **kwargs
        )
        return sim_season

    potemkin_dataset.add_sim_season(name, wrapper_f)

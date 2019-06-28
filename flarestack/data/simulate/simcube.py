from flarestack.data.public import ps_3_year
from flarestack.data.simulate import SimSeason, SimDataset

class SimCubeSeason(SimSeason):

    def generate_sim_data(self):
        return None



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

nicecube_dataset = SimDataset()

for (name, season) in ps_3_year.get_seasons().items():

    def wrapper_f(mjd_start, mjd_end, bkg_flux_norm, bkg_e_pdf_dict, **kwargs):
        sim_season = SimCubeSeason(
            season_name=name,
            sample_name="SimCube",
            pseudo_mc_path=season.pseudo_mc_path,
            mjd_start=mjd_start,
            mjd_end=mjd_end,
            bkg_flux_norm=bkg_flux_norm,
            bkg_e_pdf_dict=bkg_e_pdf_dict
        )
        return sim_season

    nicecube_dataset.add_sim_season(name, wrapper_f)

nicecube_season = nicecube_dataset.set_sim_params(
    name="IC86-2012",
    mjd_start=55000.,
    mjd_end=55100.,
    bkg_flux_norm=1.,
    bkg_e_pdf_dict=e_pdf_dict
)

print(nicecube_dataset.get_seasons())



# nicecube_10year = SimCubeSeason(0, 100, 1., e_pdf_dict)


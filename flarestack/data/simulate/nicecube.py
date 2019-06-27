from flarestack.data.public import ic_sim_seasons
from flarestack.data.simulate import generate_sim_season_class


print(ic_sim_seasons)
NiceCubeSeason = ic_sim_seasons['IC86-2012']

e_pdf_dict = {
    "name": "PowerLaw",
    "gamma": 3.7
}
nicecube_10year = NiceCubeSeason(0, 100, 1., e_pdf_dict)


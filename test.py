import numpy as np
from core.injector import Injector
from core.llh import LLH
from IceCube_data.pointsource_7year import ps_7year

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Jetted_TDE_catalogue.npy"

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Individual_TDEs/Swift J1644+57.npy"

old_sources = np.load(source_path)

sources = np.empty(
    len(old_sources), dtype=[
        ("ra", np.float), ("dec", np.float),
        ("flux", np.float),
        # ("n_exp", np.float),
        # ("weight", np.float), ("weight_acceptance", np.float),
        # ("weight_time", np.float),
        ("weight_distance", np.float),
        ("Ref Time (MJD)", np.float),
        ("distance", np.float), ('Name', 'a30'),
        ])

for x in ["ra", "dec", "distance"]:
    sources[x] = old_sources[x]

sources["Name"] = old_sources["name"]
sources["Ref Time (MJD)"] = old_sources["discoverydate_mjd"]
sources["flux"] = old_sources["flux"] * 10

sources["weight_distance"] = sources["distance"] ** -2
sources["weight_distance"] *= 1. / np.sum(sources["weight_distance"])

print sources, sources.dtype.names

injectors = dict()
llhs = dict()

for season in ps_7year:

    injection_energy = {
        "Name": "Power Law",
        "Gamma": 1.9,
        # "E Min": 10000
    }

    injection_time = {
        "Name": "Box",
        "Pre-Window": 30,
        "Post-Window": 100
    }

    kwargs = {
        "Injection Energy PDF": injection_energy,
        "Injection Time PDF": injection_time,
    }

    injectors[season["Name"]] = Injector(season, sources, **kwargs)

    kwargs = {
        "LLH Energy PDF": injection_energy,
        "LLH Time PDF": injection_time
    }

    llhs[season["Name"]] = LLH(season, sources, **kwargs)


for i in range(1):
    for season in ps_7year:
        dataset = injectors[season["Name"]].create_dataset()
        cut_dataset = llhs[season["Name"]].select_coincident_data(dataset, sources)

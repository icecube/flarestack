import numpy as np
from core.minimisation import MinimisationHandler
from IceCube_data.pointsource_7year import ps_7year

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Jetted_TDE_catalogue.npy"

# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Dai_Fang_TDE_catalogue.npy"

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Individual_TDEs/Swift J1644+57.npy"

old_sources = np.load(source_path)

# print old_sources

sources = np.empty_like(old_sources, dtype=[
    ("ra", np.float), ("dec", np.float),
    ("injection flux", np.float),
    ("llh flux", np.float),
    # ("n_exp", np.float),
    ("weight", np.float),
    # ("weight_acceptance", np.float),
    # ("weight_time", np.float),
    ("weight_distance", np.float),
    ("Ref Time (MJD)", np.float),
    ("Start Time (MJD)", np.float),
    ("End Time (MJD)", np.float),
    ("distance", np.float), ('Name', 'a30'),
])

for x in ["ra", "dec", "distance", "weight", "Start Time (MJD)",
          "End Time (MJD)"]:
    sources[x] = old_sources[x]

sources["Name"] = old_sources["name"]
sources["Ref Time (MJD)"] = old_sources["discoverydate_mjd"]
sources["llh flux"] = old_sources["flux"]

sources["weight"] = np.ones_like(old_sources["flux"])

sources["weight_distance"] = sources["distance"] ** -2

sources["injection flux"] = old_sources["flux"] * 100

injectors = dict()
llhs = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 1.9,
    # "E Min": 10000
}

injection_time = {
    "Name": "Box",
    "Pre-Window": 20.,
    "Post-Window": 30.
}

# injection_time = {
#     "Name": "Steady"
# }

# llh_time = {
#     "Name": "FixedBox"
# }

llh_time = injection_time

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
}

llh_energy = injection_energy

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    # "Fit Gamma?": True,
    # "Fit Weights?": True,
    # "Flare Search?": True
}

mh = MinimisationHandler(ps_7year, sources, inj_kwargs, llh_kwargs)

mh.run_trials(200)

# bkg_ts = mh.bkg_trials()

# bkg_median = np.median(bkg_ts)
# bkg_median = 0.0
#
# for scale in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
#
#     print "Scale", scale
#     ts = np.array(mh.run_trials(200, scale=scale)[0])
#
#     frac_over_median = np.sum(ts > bkg_median) / float(len(ts))
#
#     print "For k=" + str(scale), "we have", frac_over_median, \
#         "overfluctuations."



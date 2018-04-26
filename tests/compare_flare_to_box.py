import numpy as np
from core.minimisation import MinimisationHandler
from core.results import ResultsHandler
from data.icecube_pointsource_7year import ps_7986
from shared import catalogue_dir

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Jetted_TDE_catalogue.npy"

# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Dai_Fang_TDE_catalogue.npy"

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Individual_TDEs/Swift J1644+57.npy"

# source_path = catalogue_dir + "single_source_dec_0.10.npy"

old_sources = np.load(source_path)

sources = np.empty_like(old_sources, dtype=[
    ("ra", np.float), ("dec", np.float),
    ("Relative Injection Weight", np.float),
    ("Ref Time (MJD)", np.float),
    ("Start Time (MJD)", np.float),
    ("End Time (MJD)", np.float),
    ("Distance", np.float), ('Name', 'a30'),
])

for x in ["ra", "dec", "Start Time (MJD)", "End Time (MJD)"]:
    sources[x] = old_sources[x]

sources["Name"] = old_sources["name"]
sources["Relative Injection Weight"] = np.ones_like(old_sources["flux"]) * 60
sources["Ref Time (MJD)"] = old_sources["discoverydate_mjd"]
sources["Distance"] = old_sources["distance"]

injectors = dict()
llhs = dict()

# Initialise Injectors/LLHs

injection_energy = {
    "Name": "Power Law",
    "Gamma": 2.0,
    # "E Min": 10000
}

injection_time = {
    "Name": "Box",
    "Pre-Window": 0.,
    "Post-Window": 5.
}

injection_time = {
    "Name": "Steady"
}

llh_time = {
    "Name": "FixedBox",
    # "Pre-Window": 300.,
    # "Post-Window": 250.
}

llh_time = injection_time

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

llh_energy = injection_energy

llh_kwargs_0 = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": False
}
llh_kwargs_1 = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": False,
    "Flare Search?": True
}

llh_kwargs_2 = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": llh_time,
    "Fit Gamma?": True,
    "Flare Search?": True
}

for i, llh_kwargs in enumerate([llh_kwargs_0, llh_kwargs_1]):

    name = "tests/TEST" + str(i) + "/"
    mh = MinimisationHandler(name, ps_7986, sources, inj_kwargs,
                             llh_kwargs)
    mh.iterate_run(scale=6., n_trials=100)

    rh = ResultsHandler(name, llh_kwargs, sources, cleanup=True)

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

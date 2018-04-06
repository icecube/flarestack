import numpy as np
import resource
from tqdm import tqdm
import scipy.optimize
from core.injector import Injector
from core.llh import LLH
from core.ts_distributions import plot_background_TS_distribution
from IceCube_data.pointsource_7year import ps_7year

# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Jetted_TDE_catalogue.npy"

# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Dai_Fang_TDE_catalogue.npy"
# #
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
        ("distance", np.float), ('Name', 'a30'),
        ])

for x in ["ra", "dec", "distance", "weight"]:
    sources[x] = old_sources[x]

sources["Name"] = old_sources["name"]
sources["Ref Time (MJD)"] = old_sources["discoverydate_mjd"]
sources["llh flux"] = old_sources["flux"]

sources["weight"] = np.ones_like(old_sources["flux"])

sources["weight_distance"] = sources["distance"] ** -2

sources["injection flux"] = old_sources["flux"] * 5

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
    "Pre-Window": 20,
    "Post-Window": 0
}

# injection_time = {
#     "Name": "Steady"
# }

injection_time = {
    "Name": "Steady"
}

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
}

llh_energy = injection_energy
llh_time = injection_time

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": injection_time,
    "Fit Gamma?": True,
    "Fit Weights?": False
}

for season in ps_7year:

    injectors[season["Name"]] = Injector(season, sources, **inj_kwargs)
    llhs[season["Name"]] = LLH(season, sources, **llh_kwargs)

p0 = [1.]
bounds = [(0, 1000.)]

if "Fit Weights?" in llh_kwargs.keys():
    if llh_kwargs["Fit Weights?"]:
        p0 = [1. for x in sources]
        bounds = [(0, 1000.) for x in sources]

if "Fit Gamma?" in llh_kwargs.keys():
    if llh_kwargs["Fit Gamma?"]:
        p0.append(2.)
        bounds.append((1., 4.))

n_trials = 1000

# for scale in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
#
#     print "Scale", scale


def run_trials(scale=1, n=n_trials):

    param_vals = [[] for x in p0]
    ts_vals = []
    flags = []

    print "Generating", n, "trials!"

    for i in tqdm(range(n)):
        llh_functions = dict()

        for season in ps_7year:
            dataset = injectors[season["Name"]].create_dataset(scale)
            llh_f = llhs[season["Name"]].create_llh_function(dataset)
            llh_functions[season["Name"]] = llh_f

        def f_final(params):

            weights = dict()

            weights_matrix = np.ones([len(ps_7year), len(sources)])

            season_weights = []
            weights = dict()

            # for source in sources:

            for i, season in enumerate(ps_7year):
                llh = llhs[season["Name"]]
                acc = llh.acceptance(sources, params)

                time_weights = []

                for source in sources:
                    start = llh.time_PDF.product_integral(season["Start (MJD)"],
                                                          source)
                    end = llh.time_PDF.product_integral(season["End (MJD)"],
                                                        source)

                    time_weights.append(end - start)

                w = acc * sources["weight_distance"] * np.array(time_weights)
                w = w[:, np.newaxis]

                # season_weights.append(np.sum(w))

                for j, ind_w in enumerate(w):
                    weights_matrix[i][j] = ind_w

                # weights[season["Name"]] = w / np.sum(w)

            weights_matrix /= np.sum(weights_matrix)

            # season_weights = np.array(season_weights) /np.sum(
            #     season_weights)

            # print "Season Weights", weights_matrix
            # print weights_matrix*params[0]
            # raw_input("prompt")

            ts_val = 0
            for i, (name, f) in enumerate(llh_functions.iteritems()):
                w = weights_matrix[i][:, np.newaxis]

                # print w
                # raw_input("prompt")

                # w /= np.sum(w)
                ts_val += f(params, w, 1.)

            return -ts_val
        #
        # print p0

        res = scipy.optimize.fmin_l_bfgs_b(
            f_final, p0, bounds=bounds, approx_grad=True)

        vals = res[0]
        ts = -res[1]
        flag = res[2]["warnflag"]

        for i, val in enumerate(vals):
            param_vals[i].append(val)
        ts_vals.append(float(ts))

        flags.append(flag)

    MemUse = str(
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1.e6)
    print 'Memory usage max: %s (Gb)' % MemUse

    for i, param in enumerate(param_vals):
        print "Parameter", i, ":", np.mean(param), np.median(param), np.std(
            param)

    print "Test Statistic", np.mean(ts_vals), np.std(ts_vals)

    print "FLAG STATISTICS:"

    for i in sorted(np.unique(flags)):
        print "Flag", i, ":", flags.count(i)

    return ts_vals, param_vals, flags


print "Generating background trials"

bkg_ts, params, flags = run_trials(1.0, 1000)

ts_array = np.array(bkg_ts)

frac = float(len(ts_array[ts_array <= 0])) / (float(len(ts_array)))

# print ts_array

print "Fraction of underfluctuations is", frac

plot_background_TS_distribution(ts_array)

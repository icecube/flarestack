import numpy as np
import resource
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import scipy.optimize
from core.injector import Injector
from core.llh import LLH
from core.ts_distributions import plot_background_TS_distribution
from IceCube_data.pointsource_7year import ps_7year

source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
              "/Input/Catalogues/Jetted_TDE_catalogue.npy"

# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Dai_Fang_TDE_catalogue.npy"
# #
# source_path = "/afs/ifh.de/user/s/steinrob/scratch/The-Flux-Evaluator__Data" \
#               "/Input/Catalogues/Individual_TDEs/Swift J1644+57.npy"

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

sources["injection flux"] = old_sources["flux"] * 10

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
    "Pre-Window": 13,
    "Post-Window": 213
}

# injection_time = {
#     "Name": "Steady"
# }

inj_kwargs = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
}

llh_energy = injection_energy
llh_time = injection_time

llh_kwargs = {
    "LLH Energy PDF": llh_energy,
    "LLH Time PDF": injection_time,
    "Fit Gamma?": False,
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


def run_trials(n=n_trials, scale=1):

    param_vals = [[] for x in p0]
    ts_vals = []
    flags = []

    print "Generating", n, "trials!"

    for i in tqdm(range(int(n))):

        f = run(scale)

        res = scipy.optimize.fmin_l_bfgs_b(
            f, p0, bounds=bounds, approx_grad=True)

        flag = res[2]["warnflag"]

        if flag > 0:
            res = scipy.optimize.brute(f, ranges=bounds, full_output=True)

        vals = res[0]
        ts = -res[1]

        for j, val in enumerate(vals):
            param_vals[j].append(val)
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


def run(scale=1):

    llh_functions = dict()

    for season in ps_7year:
        dataset = injectors[season["Name"]].create_dataset(scale)
        llh_f = llhs[season["Name"]].create_llh_function(dataset)
        llh_functions[season["Name"]] = llh_f

    def f_final(params):

        weights_matrix = np.ones([len(ps_7year), len(sources)])

        for i, season in enumerate(ps_7year):
            llh = llhs[season["Name"]]
            acc = llh.acceptance(sources, params)

            time_weights = []

            for source in sources:

                time_weights.append(llh.time_PDF.effective_injection_time(
                    source))

            w = acc * sources["weight_distance"] * np.array(time_weights)

            w = w[:, np.newaxis]

            for j, ind_w in enumerate(w):
                weights_matrix[i][j] = ind_w

        weights_matrix /= np.sum(weights_matrix)

        ts_val = 0
        for i, (name, f) in enumerate(llh_functions.iteritems()):
            w = weights_matrix[i][:, np.newaxis]

            season_weight = np.sum(w) * float(len(ps_7year))

            # print np.sum(w)

            # print season_weight
            # raw_input("prompt")

            ts_val += f(params, w, 1) * season_weight

        return -ts_val

    return f_final

def scan_likelihood(scale=1):

    f = run(scale)

    n_range = np.linspace(1, 200, 1e4)
    y = []

    for n in tqdm(n_range):
        new = f([n])
        try:
            y.append(new[0][0])
        except IndexError:
            y.append(new)

    plt.figure()
    plt.plot(n_range, y)
    plt.savefig("llh_scan.pdf")
    plt.close()

    min_y = np.min(y)

    print "Minimum value of", min_y,

    min_index = y.index(min_y)

    min_n = n_range[min_index]

    print "at", min_n

    l_y = np.array(y[:min_index])

    try:
        l_y = min(l_y[l_y > (min_y + 0.5)])

        l_lim = n_range[y.index(l_y)]

    except ValueError:
        l_lim = 0

    u_y = np.array(y[min_index:])

    u_y = min(u_y[u_y > (min_y + 0.5)])

    u_lim = n_range[y.index(u_y)]

    print "One Sigma interval between", l_lim, "and", u_lim


def bkg_trials():
    print "Generating background trials"

    bkg_ts, params, flags = run_trials(1000, 0.0)

    ts_array = np.array(bkg_ts)

    frac = float(len(ts_array[ts_array <= 0])) / (float(len(ts_array)))

    print "Fraction of underfluctuations is", frac

    plot_background_TS_distribution(ts_array)

run_trials(200)
scan_likelihood()
bkg_trials()

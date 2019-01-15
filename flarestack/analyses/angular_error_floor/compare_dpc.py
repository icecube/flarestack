from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC86_1_dict
from flarestack.core.minimisation import MinimisationHandler
from flarestack.core.results import ResultsHandler
from flarestack.cluster import run_desy_cluster as rd
from flarestack.shared import flux_to_k, plot_output_dir, scale_shortener, \
    make_analysis_pickle
from flarestack.utils.reference_sensitivity import reference_sensitivity
import matplotlib.pyplot as plt
import json
import numpy as np

basename = "analyses/angular_error_floor/compare_dpc/"

gamma = 3.0

injection_energy = {
    "Name": "Power Law",
    "Gamma": gamma,
}

injection_time = {
    "Name": "Steady"
}

inj_dict = {
    "Injection Energy PDF": injection_energy,
    "Injection Time PDF": injection_time,
    "Poisson Smear?": True,
}

sin_decs = np.linspace(0.90, -0.90, 13)
# sin_decs = [-0.5, 0.0, 0.5]
res_dict = dict()


for pull_corrector in ["no_pull", "median_1d", "median_0d_e"]:

    root_name = basename + pull_corrector + "/"

    for floor in ["no_floor"]:
        seed_name = root_name + floor + "/"

        key = pull_corrector + "/" + floor

        config_mh = []

        for sin_dec in sin_decs:

            name = seed_name + "sindec=" + '{0:.2f}'.format(sin_dec) + "/"

            llh_dict = {
                "name": "standard",
                "LLH Energy PDF": injection_energy,
                "LLH Time PDF": injection_time,
                "pull_name": pull_corrector,
                "floor_name": floor
            }

            scale = flux_to_k(reference_sensitivity(sin_dec, gamma)) * 10

            mh_dict = {
                "name": name,
                "mh_name": "fixed_weights",
                "datasets": [IC86_1_dict],
                "catalogue": ps_catalogue_name(sin_dec),
                "llh_dict": llh_dict,
                "inj kwargs": inj_dict,
                "n_trials": 20,
                "n_steps": 10,
                "scale": scale
            }

            pkl_file = make_analysis_pickle(mh_dict)

            # rd.submit_to_cluster(pkl_file, n_jobs=500)

            mh = MinimisationHandler.create(mh_dict)
            mh.iterate_run(scale=scale, n_steps=5, n_trials=100)

            config_mh.append(mh_dict)

        res_dict[key] = config_mh

rd.wait_for_cluster()
sens_dict = dict()
bias_dict = dict()

plt.figure()
ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=3, rowspan=3)

for (config, mh_list) in res_dict.iteritems():

    sens = []

    biases = []

    for mh_dict in mh_list:
        rh = ResultsHandler(mh_dict)

        max_scale = scale_shortener(max([float(x) for x in rh.results.keys()]))
        sens.append(rh.sensitivity)

        fit = rh.results[max_scale]["Parameters"]["n_s"]
        inj = rh.inj[max_scale]["n_s"]
        med_bias = np.median(fit)/inj

        biases.append(med_bias)

    ax1.plot(sin_decs, sens, label=config)
    sens_dict[config] = sens
    bias_dict[config] = biases

# ax1.set_ylim(ymin=1.e-13, ymax=1.e-10)
ax1.grid(True, which='both')
ax1.semilogy(nonposy='clip')

ax1.set_ylabel(r"Flux Strength [ GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ ]")
ax1.legend()
plt.title(r'Comparison of Static Pull Corrections with $E^{-'
          + str(gamma) + "}$")

ax2 = plt.subplot2grid((4, 1), (3, 0), colspan=3, rowspan=1, sharex=ax1)

for (config, sens) in sens_dict.iteritems():
    ax2.plot(sin_decs, np.array(sens)/np.array(sens_dict["no_pull/no_floor"]))

ax2.set_xlabel(r"sin($\delta$)")
ax2.set_ylabel("Ratio")

ax1.set_xlim(xmin=-1.0, xmax=1.0)
ax2.grid(True)
xticklabels = ax1.get_xticklabels()
plt.setp(xticklabels, visible=False)
plt.subplots_adjust(hspace=0.001)

savepath = plot_output_dir(basename) + "comparison_sensitivity.pdf"

print "Saving to", savepath
plt.savefig(savepath)
plt.close()

plt.figure()
for (config, biases) in bias_dict.iteritems():
    plt.plot(sin_decs, biases, label=config)

plt.axhline(1.0, linestyle=":")
plt.xlabel(r"sin($\delta$)")
plt.ylabel(r"Median Bias in $n_{s}$")
plt.legend()
plt.title(r'Median Bias with $E^{-'+ str(gamma) + "}$")
savepath = plot_output_dir(basename) + "comparison_bias.pdf"
print "Saving to", savepath
plt.savefig(savepath)
plt.close()


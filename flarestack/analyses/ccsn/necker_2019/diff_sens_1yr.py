import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import raw_output_dir
from flarestack.utils.prepare_catalogue import ps_catalogue_name
from flarestack.utils.custom_dataset import custom_dataset
from flarestack.data.icecube import ps_v004_p00
from flarestack.cluster.submitter import Submitter
from flarestack.core.results import ResultsHandler, OverfluctuationError
from flarestack.shared import cache_dir, flux_to_k, scale_shortener


logger = logging.getLogger("flarestack.analyses.ccsn.necker_2019.diff_sens_1yr")


def get_raw_ref_sens(sindec):
    fn = '~/Downloads/Unknown-3'
    logger.debug(f"loading {fn}")
    ref_sens = pd.read_csv(fn, header=[0, 1])
    col_mask = [float(ii[0].split("sindec")[1]) == sindec if "sindec" in ii[0] else False for ii in ref_sens.columns]
    x_col = ref_sens.columns[col_mask]
    y_col = ref_sens.columns[np.where(col_mask)[0] + 1]
    logger.debug(f"interpolating {x_col} and {y_col}")
    return ref_sens[x_col].values.flatten(), ref_sens[y_col].values.flatten(),


def get_ref_sens(sindec):
    x, y = get_raw_ref_sens(sindec)
    f = interp1d(x, y, fill_value="extrapolate")
    return f


def get_scale(sindec, e):
    logger.info(f"getting scale for gamma=2.0, sindec={sindec:.2f}, e={e:.2f}")
    f = get_ref_sens(sindec)
    k = flux_to_k(f(e)) * 2

    if (sindec == -0.5) & (e < 1e5):
        k *= 2

    logger.debug(f"scale={k:.2e}")
    return k


def get_mh_dict(sindec, gamma, emin, emax):

    ic86_2 = ps_v004_p00.get_seasons("IC86_2")

    time_pdf_dict = {"time_pdf_name": "steady"}

    cat_name = ps_catalogue_name(sindec)
    cat = np.load(cat_name)
    dataset = custom_dataset(ic86_2, cat, time_pdf_dict)

    llh_dict = {
        "llh_name": "standard",
        "llh_energy_pdf": {"energy_pdf_name": "power_law"},
        "llh_sig_time_pdf": time_pdf_dict,
        "llh_bkg_time_pdf": {"time_pdf_name": "steady"}
    }

    if sindec == -0.5:
        llh_dict["llh_energy_pdf"]["e_min_gev"] = emin
        llh_dict["llh_energy_pdf"]["e_max_gev"] = emax

    inj_dict = {
        "injection_energy_pdf": {
            "energy_pdf_name": "power_law",
            "gamma": gamma,
            "e_min_gev": emin,
            "e_max_gev": emax
        },
        "injection_sig_time_pdf": time_pdf_dict,
        "poisson_smear_bool": True,
    }

    mh_dict = {
        "name": raw_output_dir + f"/ic86_2_diff_ps_sens_separate_bkg/gamma{gamma:.2f}/sindec{sindec:.2f}/{emin:.2f}GeV_to_{emax:.2f}GeV",
        "mh_name": "fixed_weights",
        "scale": np.nan,
        "gamma": gamma,
        "dataset": dataset,
        "catalogue": cat_name,
        "inj_dict": inj_dict,
        "llh_dict": llh_dict,
        "n_trials": 1000,
        "n_steps": 10,
        "background_ntrials_factor": 0,
    }

    return mh_dict


def run_signal_trials(sindec, gamma, emin, emax):
    mh_dict = get_mh_dict(sindec, gamma, emin, emax)
    mh_dict["scale"] = get_scale(sindec, (emin + emax) / 2)
    logger.info(f"running trials for {mh_dict['name']}")
    s = Submitter(mh_dict, use_cluster=False, do_sensitivity_scale_estimation=False, n_cpu=6)
    s.analyse(do_disc=False)


def run_background_trials(sindec):
    mh_dict = get_mh_dict(sindec, gamma=np.nan, emin=np.nan, emax=np.nan)
    mh_dict["fixed_scale"] = 0
    mh_dict["trials"] = 10000
    logger.info(f"running background trials trials for {mh_dict['name']}")
    s = Submitter(mh_dict, use_cluster=False, do_sensitivity_scale_estimation=False, n_cpu=6)
    s.analyse(do_disc=False)


def calculate_sens(sindec, gamma, emin, emax, allow_extrapolated=False):
    mh_dict = get_mh_dict(sindec, gamma, emin, emax)
    mh_dict["allow_extrapolated_sensitivity"] = allow_extrapolated
    logger.info(f"calculating sens for {mh_dict['name']}")

    bkg_mh_dict = get_mh_dict(sindec, np.nan, np.nan, np.nan)

    try:
        bkg_rh = ResultsHandler(bkg_mh_dict, do_sens=False, do_disc=False)
    except FileNotFoundError:
        run_background_trials(sindec)
        bkg_rh = ResultsHandler(bkg_mh_dict, do_sens=False, do_disc=False)

    def _calc():
        rh = ResultsHandler(mh_dict, do_disc=False, do_sens=False)
        rh.results[scale_shortener(0)] = bkg_rh.results[scale_shortener(0)]
        rh.inj[scale_shortener(0)] = bkg_rh.results[scale_shortener(0)]
        rh.find_sensitivity()
        return rh

    try:

        try:
            rh = _calc()
        except FileNotFoundError:
            run_signal_trials(sindec, gamma, emin, emax)
            rh = _calc()

        return rh.sensitivity, list(rh.sensitivity_err)

    except OverfluctuationError as e:
        logger.error(f"{mh_dict['name']}: {e}")


def get_results(gamma):
    logger.info(f"getting results for gamma={gamma:.2f}")
    result_filename = os.path.join(cache_dir, raw_output_dir, "ic86_2_diff_ps_sens_separate_bkg", f"gamma{gamma:.2f}.json")
    d = os.path.dirname(result_filename)
    if not os.path.isdir(d):
        os.makedirs(d)

    if os.path.isfile(result_filename):
        logger.info(f"loading {result_filename}")
        with open(result_filename, "r") as f:
            results = json.load(f)

    else:
        results = dict()

    sindecs = [-0.5, 0, 0.5]

    update = False

    for sindec in sindecs:
        logger.info(f"sindec={sindec:.2f}")
        energies = np.logspace(2, 7, 11) if sindec >= 0 else np.logspace(4, 7, 7)

        if str(sindec) not in results:
            results[str(sindec)] = dict()

        for emin, emax in zip(energies[:-1], energies[1:]):
            logger.info(f"{emin:.2f}-{emax:.2f} GeV")

            if str(emin) not in results[str(sindec)]:
                results[str(sindec)][str(emin)] = calculate_sens(sindec, gamma, emin, emax)
                update = True

    if update:
        logger.info(f"saving to {result_filename}")
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=4)

    return results


def make_plot(gamma, plot_ref_sens=False):
    result = get_results(gamma)

    sindec_color = {
        0.5: "C0",
        0: "C1",
        -0.5: "C2"
    }

    sindec_ls = {
        0.5: "-",
        0: "--",
        -0.5: ":"
    }

    fig, ax = plt.subplots(figsize=(3.5 * 1.61803, 3.5))

    for sindec, energy_dict in result.items():
        energies = np.array(list(energy_dict.keys())).astype(float)
        sens = np.array([l[0] if l is not None else None for l in energy_dict.values()]).astype(float)

        sort_inds = np.argsort(energies)
        sorted_energies = np.array(list(energies[sort_inds]) + [1e7])

        ax.step(
            sorted_energies,
            list(sens[sort_inds]) + [sens[sort_inds][-1]],
            color=sindec_color[float(sindec)],
            ls=sindec_ls[float(sindec)],
            label=f"$\sin(\delta)=${float(sindec):.2f}",
            where="post"
        )

        if plot_ref_sens:

            x, y = get_raw_ref_sens(float(sindec))
            nan_m = ~np.isnan(x) & ~np.isnan(y)
            ax.plot(x[nan_m], y[nan_m],
                       color=sindec_color[float(sindec)],
                       alpha=0.5,
                       label=f"ref raw $\sin(\delta)=${float(sindec):.2f}",
                       marker=""
                       )

            f = get_ref_sens(float(sindec))
            ax.step(
                sorted_energies,
                f(list((sorted_energies[:-1] + sorted_energies[1:]) / 2) + [sorted_energies[-1]]),
                color=sindec_color[float(sindec)],
                ls=sindec_ls[float(sindec)],
                alpha=0.5,
                label=f"ref $\sin(\delta)=${float(sindec):.2f}",
                where="post"
            )

    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$\frac{E^2 \mathrm{d}N}{\mathrm{d}A \, \mathrm{d}E \, \mathrm{d}t}$ [GeV cm$^{-2}$ s$^{-1}$]")
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("flarestack").setLevel("DEBUG")
    make_plot(2.0, plot_ref_sens=True)

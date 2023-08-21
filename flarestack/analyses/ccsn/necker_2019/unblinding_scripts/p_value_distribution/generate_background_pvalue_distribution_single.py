import logging, os, pickle, argparse
import numpy as np

from flarestack.shared import analysis_pickle_path, storage_dir
from flarestack.core.unblinding import create_unblinder
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import (
    sn_cats,
    sn_time_pdfs,
    raw_output_dir,
)

# from flarestack.analyses.ccsn.necker_2019.unblinding_scripts.p_value_distribution.generate_background_pvalue_distributions import \
#     base_raw, gammas, p_value_filename_single


logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")
logger = logging.getLogger("main")

# name of this file
single_filename = os.path.abspath(__file__)

# -------------------------------------------------------------------------------------
# START define some parameters

# LLH Energy PDF
llh_energy = {
    "energy_pdf_name": "power_law",
}

N_trials = 500

cluster = False

# Spectral indices to loop over
gammas = [2.0, 2.5]

# minimizer to use
mh_name = "fit_weights"

# base name
base_raw = raw_output_dir + f"/calculate_sensitivity_ps-v002p03/{mh_name}"

# p-values storage directory and file
p_value_directory = os.path.join(storage_dir, base_raw, "pvalues")
p_value_filename = os.path.join(storage_dir, base_raw, "pvalues.pkl")

# END define some parameters
# -------------------------------------------------------------------------------------


def p_value_filename_single(seed):
    """gives a filename in the pvalues directory depending on the seed that is used"""
    return os.path.join(p_value_directory, f"{seed}.pkl")


# -----------------––––---------------------------------------------
# START define function that runs background p-value calculation
# -----------------––––---------------------------------------------


def run_background_pvalue_trials(seed=None, ntrials=1):
    """
    Run background trials and calculate the p-values
    :param seed: int, seed for dataset simulation
    :param ntrials: int, number of trials to run
    :return: dictionary containing the trial results
    """

    # setting the random seed ensures that the dataset produced by the Unblinder
    # will always produce the same dataset
    rng = np.random.default_rng(seed=seed)

    # generating an array of seeds that will be used if more than one trial is run
    seeds = rng.integers(0, 2**30, ntrials)

    pdf_full_res = dict()

    for pdf_type in ["box", "decay"]:
        # base name
        raw = f"{base_raw}/{pdf_type}/"

        # set up emtpy dictionary to store the minimizer information in
        full_res = dict()

        # loop over SN catalogues
        use_cats = sn_cats if pdf_type == "box" else ["IIn", "IIP"]
        for cat in use_cats:
            name = raw + cat + "/"

            # set up empty results dictionary for this catalogue
            cat_res = dict()

            # get the time pdfs for this catalogue
            time_pdfs = sn_time_pdfs(cat, pdf_type=pdf_type)

            # Loop over time PDFs
            for llh_time in time_pdfs:
                # set up an empty results array for this time pdf
                time_res = dict()

                logging.debug(f"time_pdf is {llh_time}")

                if pdf_type == "box":
                    time_key = str(llh_time["post_window"] + llh_time["pre_window"])
                    # pdf_time = float(time_key) if llh_time['pre_window'] == 0 else - float(time_key)

                else:
                    time_key = str(llh_time["decay_time"])

                time_name = name + time_key + "/"

                # Loop over spectral indices
                for gamma in gammas:
                    # load the background trials
                    if (pdf_type == "decay") and (gamma == 2.0):
                        gamma_str = "2"
                    else:
                        gamma_str = str(gamma)

                    full_name = time_name + gamma_str + "/"

                    mh_dict_path = analysis_pickle_path(name=full_name)
                    logger.debug(f"loading analysis pickle from {mh_dict_path}")
                    with open(mh_dict_path, "rb") as f:
                        mh_dict = pickle.load(f)

                    unblind_dict = dict(mh_dict)
                    unblind_dict["name"] += "_create_backround_pvalue_distr"
                    unblind_dict["background_ts"] = mh_dict["name"]

                    ub = create_unblinder(
                        unblind_dict,
                        mock_unblind=True,
                        disable_warning=True,
                        seed=seeds[0],
                    )
                    bkg_ts_array = ub.bkg_ts_array
                    p_values = [ub.raw_pre_trial_pvalue]
                    logger.info(f"Raw p-value is {ub.raw_pre_trial_pvalue}")

                    for s in seeds[1:]:
                        int_seed = s
                        logger.debug(f"seed is {int_seed}")
                        res_dict = ub.simulate_and_run(0, int_seed)
                        ts = res_dict["TS"]

                        raw_p_value = len(bkg_ts_array[bkg_ts_array > ts]) / len(
                            bkg_ts_array
                        )
                        logger.info(f"Raw p-value is {raw_p_value}")
                        p_values.append(raw_p_value)

                    time_res[gamma] = p_values
                cat_res[time_key] = time_res
            full_res[cat] = cat_res
        pdf_full_res[pdf_type] = full_res

    return pdf_full_res


# -----------------––––---------------------------------------------
# END define function that runs background p-value calculation
# -----------------––––---------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrials", type=int, default=1)
    cfg = parser.parse_args()

    seed = np.random.default_rng().integers(low=2**30)
    pdf_full_res = run_background_pvalue_trials(seed, cfg.ntrials)

    with open(p_value_filename_single(seed), "wb") as f:
        pickle.dump(pdf_full_res, f)

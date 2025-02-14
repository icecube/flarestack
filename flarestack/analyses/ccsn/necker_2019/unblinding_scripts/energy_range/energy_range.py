"""Script to calculate the sensitivity and discovery potential for CCSNe.
"""
import numpy as np
import math, pickle, os, logging, time, shutil
from flarestack.core.results import ResultsHandler, OverfluctuationError
from flarestack.core.experimental_results import ExperimentalResultHandler
from flarestack.data.icecube import ps_v002_p03
from flarestack.shared import (
    flux_to_k,
    scale_shortener,
    storage_dir,
    pickle_dir,
    inj_param_dir,
)
from flarestack.icecube_utils.reference_sensitivity import reference_sensitivity
from flarestack.analyses.ccsn.necker_2019.ccsn_helpers import (
    sn_cats,
    updated_sn_catalogue_name,
    sn_time_pdfs,
    raw_output_dir,
    pdf_names,
)
from flarestack.cluster import analyse
from flarestack.cluster.run_desy_cluster import wait_for_cluster
from flarestack.utils.custom_dataset import custom_dataset

# Set Logger Level

start = time.time()

logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")

# LLH Energy PDF
llh_energy = {
    "energy_pdf_name": "power_law",
}

cluster = 350

# Spectral indices to loop over
gammas = [2.0, 2.5]

# minimizer to use
mh_name = "fit_weights"

# base name
base_raw = raw_output_dir + f"/calculate_sensitivity_ps-v002p03/{mh_name}"

# filename for results
energy_range_sens_filename = os.path.join(storage_dir, base_raw, "energy_range_res.pkl")
energy_range_filename = os.path.join(storage_dir, base_raw, "energy_range.pkl")

# define ranges of min/max energy to test
# e_range = np.logspace(2, 7, 12)
energy_ranges = {
    2.0: {"e_min_gev": np.logspace(3, 4.5, 6), "e_max_gev": np.logspace(5.5, 7, 6)},
    2.5: {"e_min_gev": np.logspace(2, 3.5, 6), "e_max_gev": np.logspace(5, 6, 6)},
}

if __name__ == "__main__":
    pdf_full_res = dict()

    # set up empty list for cluster job IDs
    job_ids = []

    for pdf_type in ["decay", "box"]:
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
            for llh_time in time_pdfs[:1]:
                # set up an empty results array for this time pdf
                time_res = dict()

                logging.debug(f"time_pdf is {llh_time}")

                if pdf_type == "box":
                    time_key = str(llh_time["post_window"] + llh_time["pre_window"])
                    pdf_time = (
                        float(time_key)
                        if llh_time["pre_window"] == 0
                        else -float(time_key)
                    )

                else:
                    time_key = str(llh_time["decay_time"])
                    pdf_time = llh_time["decay_time"] / 364.25

                pdf_name = pdf_names(pdf_type, pdf_time)
                cat_path = updated_sn_catalogue_name(cat)
                logging.debug("catalogue path: " + str(cat_path))

                # load catalogue and select the closest source
                # that serves for estimating a good injection scale later
                catalogue = np.load(cat_path)
                logging.debug("catalogue dtype: " + str(catalogue.dtype))
                closest_src = np.sort(catalogue, order="distance_mpc")[0]

                time_name = name + time_key + "/"

                # set up the likelihood dictionary
                llh_dict = {
                    "llh_name": "standard",
                    "llh_energy_pdf": llh_energy,
                    "llh_sig_time_pdf": llh_time,
                    "llh_bkg_time_pdf": {"time_pdf_name": "steady"},
                }

                # set up an injection dictionary which will be equal to the time pdf dictionary
                injection_time = llh_time

                # Loop over spectral indices
                for gamma in gammas:
                    # if (gamma == 2.) and (pdf_type == 'decay'):
                    #     gamma_str = '2'
                    # else:
                    #     gamma_str = str(gamma)

                    full_name = time_name + str(gamma) + "/"

                    length = float(time_key)

                    # try to estimate a good scale based on the sensitivity from the 7-yr PS sensitivity
                    # at the declination of the closest source
                    scale = (
                        0.5
                        * (
                            flux_to_k(
                                reference_sensitivity(
                                    np.sin(closest_src["dec_rad"]), gamma=gamma
                                )
                                * 40
                                * math.sqrt(float(len(catalogue)))
                            )
                            * 200.0
                        )
                        / length
                    )

                    # in some cases the sensitivity is outside the tested range
                    # to get a good sensitivity, adjust the scale in these cases
                    if pdf_type == "box":
                        if cat == "IIP":
                            scale *= 0.2
                        if (cat == "IIn") and (gamma == 2.5):
                            scale *= 1
                        if cat == "Ibc":
                            scale *= 0.5
                        if cat == "IIn" and gamma == 2.5 and pdf_time == 1000:
                            scale *= 1.4
                        if (cat == "IIP") and (gamma == 2.5):
                            scale *= 0.3

                    if pdf_type == "decay":
                        scale /= 2000

                        if (gamma == 2.5) and (cat == "IIn"):
                            scale *= 1.8
                        if (gamma == 2.5) and (cat == "IIP"):
                            scale *= 0.15
                        if (gamma == 2.0) and (cat == "IIP"):
                            scale *= 0.5
                        if cat == "IIn":
                            scale *= 5
                        if length > 700:
                            scale *= 2

                    ebound_res = dict()
                    for ebound_key, ebound_range in energy_ranges[gamma].items():
                        erange_res = dict()
                        for ebound in ebound_range:
                            # set up an injection dictionary and set the desired spectral index
                            injection_energy = dict(llh_energy)
                            injection_energy["gamma"] = gamma
                            injection_energy[ebound_key] = ebound

                            inj_dict = {
                                "injection_energy_pdf": injection_energy,
                                "injection_sig_time_pdf": injection_time,
                                "poisson_smear_bool": True,
                            }

                            full_e_name = f"{full_name}/{ebound_key}/{ebound:.4f}/"

                            # set up the final minimizer dictionary
                            mh_dict = {
                                "name": full_e_name,
                                "mh_name": mh_name,
                                "dataset": custom_dataset(
                                    ps_v002_p03, catalogue, llh_dict["llh_sig_time_pdf"]
                                ),
                                "catalogue": cat_path,
                                "inj_dict": inj_dict,
                                "llh_dict": llh_dict,
                                "scale": scale,
                                "n_trials": 700 / cluster if cluster else 10,
                                "n_steps": 10,
                                "allow_extrapolated_sensitivity": False,
                                # this determines how many background trials are run
                                "background_ntrials_factor": 0,
                            }
                            erange_res[ebound] = mh_dict

                            # call the main analyse function
                            job_id = None
                            # if (cat == 'IIn') and (gamma == 2.5) and (pdf_type == 'box'):
                            #     job_id = analyse(mh_dict,
                            #                      cluster=True if cluster else False,
                            #                      n_cpu=1 if cluster else 15,
                            #                      n_jobs=cluster,
                            #                      h_cpu='01:59:59',
                            #                      remove_old_results=True)
                            job_ids.append(job_id)

                        ebound_res[ebound_key] = erange_res

                    time_res[gamma] = ebound_res

                cat_res[time_key] = time_res

            full_res[cat] = cat_res

        pdf_full_res[pdf_type] = full_res

    # Wait for cluster. If there are no cluster jobs, this just runs
    if cluster and np.any(job_ids):
        logging.info(f"waiting for jobs {job_ids}")
        wait_for_cluster(job_ids)

    # set up a dictionary to store the results
    energy_range_sensitivities = dict()

    for pdf_type, full_res in pdf_full_res.items():
        pdf_type_sens = dict()

        for cat, cat_res in full_res.items():
            cat_sens = dict()

            for time_key, time_res in cat_res.items():
                time_sens = dict()

                for gamma, ebound_res in time_res.items():
                    gamma_sens = dict()

                    for ebound_key, erange_res in ebound_res.items():
                        ebound_key_sens = dict()

                        for ebound, mh_dict in erange_res.items():
                            wrong_name = (
                                raw_output_dir
                                + f"{base_raw}/{pdf_type}/{cat}/{time_key}/{gamma}/{ebound_key}/{ebound:.4f}/"
                            )
                            right_name = mh_dict["name"]

                            for d in [inj_param_dir, pickle_dir]:
                                wrong_d = os.path.join(d, wrong_name)
                                right_d = os.path.join(d, right_name)
                                logging.debug(f"copying from {wrong_d} to {right_d}")
                                try:
                                    shutil.copytree(wrong_d, right_d)
                                except (FileExistsError, FileNotFoundError):
                                    pass

                            # load the background trials
                            if (pdf_type == "decay") and (gamma == 2.0):
                                gamma_str = "2"
                            else:
                                gamma_str = str(gamma)

                            sens_name = (
                                f"{base_raw}/{pdf_type}/{cat}/{time_key}/{gamma_str}"
                            )
                            sens_mh_dict = dict(mh_dict)
                            sens_mh_dict["name"] = sens_name
                            logging.info("----- loading sensitivity results -------")
                            sens_rh = ResultsHandler(
                                sens_mh_dict, do_disc=False, do_sens=False
                            )
                            # sens_rh = ExperimentalResultHandler(sens_mh_dict, do_sens=False, do_disc=False)
                            bkg_res = sens_rh.results[scale_shortener(0.0)]

                            # load the signal trials
                            # rh = ResultsHandler(mh_dict, do_disc=False, do_sens=False)
                            try:
                                logging.info(
                                    "----------- loading energy range calculations ------------"
                                )
                                # rh = ExperimentalResultHandler(mh_dict, do_sens=False, do_disc=False)
                                rh = ResultsHandler(
                                    mh_dict, do_sens=False, do_disc=False
                                )

                                # insert the background trials
                                logging.debug(
                                    "setting background TS distribution from sensitivity results"
                                )
                                rh.inj["0"] = sens_rh.inj["0"]
                                rh.results[scale_shortener(0.0)] = bkg_res

                                # try to find sensitivity
                                rh.find_sensitivity()

                                ebound_key_sens[ebound] = (
                                    rh.sensitivity,
                                    rh.sensitivity_err,
                                )

                            except OverfluctuationError as e:
                                logging.warning(
                                    f"OverfluctuationError for {ebound}: {e}!"
                                )
                                ebound_key_sens[ebound] = tuple()

                            except FileNotFoundError as e:
                                logging.warning(f"FileNotFoundError for {ebound}: {e}!")
                                ebound_key_sens[ebound] = tuple()

                        gamma_sens[ebound_key] = ebound_key_sens
                    time_sens[gamma] = gamma_sens
                cat_sens[time_key] = time_sens
            pdf_type_sens[cat] = cat_sens
        energy_range_sensitivities[pdf_type] = pdf_type_sens

    d = os.path.dirname(energy_range_sens_filename)
    if not os.path.isdir(d):
        os.makedirs(d)

    logging.debug(f"saving under {energy_range_sens_filename}")
    with open(energy_range_sens_filename, "wb") as f:
        pickle.dump(energy_range_sensitivities, f)

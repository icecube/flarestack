import numpy as np
import os
import logging
import pickle
from flarestack.utils.prepare_catalogue import single_source
from flarestack.shared import fs_scratch_dir


logger = logging.getLogger(__name__)

data_dir = os.path.dirname(os.path.realpath(__file__))
vary_sindecs_data_dir = f"{data_dir}/vary_sindecs"
if not os.path.exists(vary_sindecs_data_dir):
    os.mkdir(vary_sindecs_data_dir)


def get_skylab_crosscheck_stacking_same_sources_res(dataset, seasons, disc=False):
    path = (
        f'{os.environ["SKYLAB_SCRATCH_DIR"]}/data/fs_crosscheck_with_same_scrambles/stacking_sensitivity_nsources_'
        f"{dataset}_"
    )
    if seasons == "all":
        path += "all"
    else:
        for s in seasons:
            path += s + "_"

    if disc:
        path += "_wdisc"

    path += "/res.pkl"
    if os.path.isfile(path):
        logger.debug(f"loading {path}")
        with open(path, "rb") as f:
            res = pickle.load(f, encoding="latin1")
        return res
    else:
        logger.warning(f"File {path} not found!")
        return


def get_skylab_crosscheck_stacking_same_sources_ts_array(
    dataset, seasons, kernel, gamma, hsphere, bgk=False, disc=False
):
    path = (
        f'{os.environ["SKYLAB_SCRATCH_DIR"]}/data/storage/fs_crosscheck_with_same_scrambles/'
        f"stacking_sensitivity_nsources_"
        f"{dataset}_"
    )
    if seasons == "all":
        path += "all"
    else:
        for s in seasons:
            path += s + "_"
    if disc:
        path += "_wdisc"
    path += f'/{kernel}/{hsphere}/{gamma:.2f}/{"bkg_" if bgk else "sensitivity" if not disc else "discovery"}trials.npy'
    # if bgk:
    #     return np.array([[0, np.load(path)[0]['TS']]])
    if os.path.isfile(path):
        logger.debug(f"loading {path}")
        res = np.load(path, allow_pickle=True, encoding="latin1")
        ns = [r[0] for r in res]
        ts = [r[-1]["TS"] for r in res]
        res_transposed = np.array([[n, t] for n, t in zip(ns, ts)])

        logger.debug(f"shape is {np.shape(res_transposed)}")
        return res_transposed
    else:
        logger.warning(f"File {path} not found!")
        return


# -------------- functions to create sources for varying declinations ------------- #


def get_original_sources(sindec, nsources):
    sources = single_source(sindec)
    src_ras = np.linspace(0, 2 * np.pi, nsources, endpoint=False)
    sources["ra_rad"][0] = src_ras[0]
    sources["source_name"] = str(src_ras[0])
    for i in range(nsources - 1):
        new_source = single_source(sindec)
        new_source["ra_rad"] = src_ras[i + 1]
        new_source["source_name"] = str(src_ras[i + 1])
        sources = np.append(sources, new_source)
    return sources


def vary_sindecs(original_sindecs, step, increase_param):
    new_sindecs = list()
    for i, orig_sindec in enumerate(original_sindecs):
        new_sindecs.append(
            orig_sindec + (-1) ** i * increase_param * step * i / len(original_sindecs)
        )
    return new_sindecs


def make_varying_sindecs_sources(sindec, step, fn, **kwargs):
    original_sources = get_original_sources(sindec, nsources=kwargs.get("nsources", 10))
    new_sindecs = vary_sindecs(
        np.sin(original_sources["dec_rad"]),
        step,
        increase_param=kwargs.get("increase_param", 0.2),
    )
    original_sources["dec_rad"] = np.arcsin(new_sindecs)
    np.save(fn, original_sources)


def get_fs_vary_sindecs_file(sindec, step, **kwargs):
    fn = f"{vary_sindecs_data_dir}/sindec{sindec:.2f}_step{step:.0f}.npy"

    if not os.path.isfile(fn) or kwargs.get("force_new_sources", False):
        logging.debug(f"making new sources for sin(dec)={sindec:.2f}, step {step:.0f}")
        make_varying_sindecs_sources(sindec, step, fn, **kwargs)

    return f"{vary_sindecs_data_dir}/sindec{sindec:.2f}_step{step:.0f}.npy"

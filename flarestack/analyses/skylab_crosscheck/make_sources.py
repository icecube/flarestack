import os
import numpy as np
import logging
from flarestack.utils.prepare_catalogue import single_source


logging.getLogger().setLevel(logging.INFO)

home_dir = os.environ["HOME"]
output_dir_for_sl_raw = home_dir + "/skylab/skylab/fs_crosscheck/data/"
output_dir_for_fs_raw = os.path.dirname(os.path.realpath(__file__)) + "/data/"

np.random.seed(1)
nsources = [1, 3, 9, 27, 81]
Nsources = max(nsources)
same_sindecs = np.linspace(-1, 1, 9)


def output_dir_for_sl(sindec=None):
    if not sindec:
        return output_dir_for_sl_raw
    else:
        return output_dir_for_sl_raw + "{:.4f}/".format(sindec)


def output_dir_for_fs(sindec=None):
    if not sindec:
        return output_dir_for_fs_raw
    else:
        return output_dir_for_fs_raw + "{:.4f}/".format(sindec)


def fs_sources(nsources, sindec=None):
    return output_dir_for_fs(sindec) + str(nsources) + "sources.npy"


def sl_sources(i):
    return sources[: nsources[i]]


if __name__ == "__main__":
    input("really continue making new sources??? ")

    # make sources with arbitrary declinations
    sindecs = np.random.uniform(-1, 1, Nsources)
    sources = single_source(sindecs[0])

    for sindec in sindecs[1:]:
        j = len(sources)
        new_source = single_source(sindec)
        new_source["ra_rad"] = float(np.random.uniform(0, 2 * np.pi, 1))
        sources = np.append(sources, new_source)
        if len(sources) != j + 1:
            raise Exception("j={0} and len(sources)={1}".format(j, len(sources)))

    for dir in [output_dir_for_fs(), output_dir_for_sl()]:
        logging.info(
            "saving {0} sources to {1}".format(len(sources), dir + "sources.npy")
        )
        np.save(dir + "sources.npy", sources)

    for n in nsources:
        logging.info(
            "saving {0} sources to {1}".format(
                len(sources[:n]), output_dir_for_fs() + str(n) + "sources.npy"
            )
        )
        np.save(output_dir_for_fs() + str(n) + "sources.npy", sources[:n])

    # make sources with the same declinations but different right ascensions
    for sindec in same_sindecs:
        sources = single_source(sindec)
        for i in range(Nsources - 1):
            new_source = single_source(sindec)
            new_source["ra_rad"] = float(np.random.uniform(0, 2 * np.pi, 1))
            sources = np.append(sources, new_source)

        for dir in [output_dir_for_fs(sindec), output_dir_for_sl(sindec)]:
            if not os.path.isdir(dir):
                os.mkdir(dir)
            logging.info(
                "saving {0} sources to {1}".format(
                    len(sources),
                    dir + "sources.npy",
                )
            )
            np.save(dir + "sources.npy", sources)

        for n in nsources:
            logging.info(
                "saving {0} sources to {1}".format(
                    len(sources[:n]), output_dir_for_fs(sindec) + str(n) + "sources.npy"
                )
            )
            np.save(output_dir_for_fs(sindec) + str(n) + "sources.npy", sources[:n])

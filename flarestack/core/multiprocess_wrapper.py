import pickle
import logging
import argparse
from flarestack.core.minimisation import MinimisationHandler, read_mh_dict
from multiprocessing import JoinableQueue, Process
import random
import numpy as np


def generate_dynamic_mh_class(mh_dict):

    # mh_dict = read_mh_dict(mh_dict)

    try:
        mh_name = mh_dict["mh_name"]
    except KeyError:
        raise KeyError("No MinimisationHandler specified.")

    # Set up dynamic inheritance

    try:
        ParentMinimisationHandler = MinimisationHandler.subclasses[mh_name]
    except KeyError:
        raise KeyError("Parent class {} not found.".format(mh_name))

    class MultiProcessingMinimisationHandler(ParentMinimisationHandler):

        def add_injector(self, season, sources):
            pass

    return MultiProcessingMinimisationHandler(mh_dict)


class MultiProcessor:
    queue = None
    results = dict()

    def __init__(self, n_cpu, **kwargs):
        self.queue = JoinableQueue()
        self.processes = [Process(target=self.run_trial, kwargs=kwargs)
                          for _ in range(int(n_cpu))]

        self.mh = MinimisationHandler.create(kwargs["mh_dict"])
        self.mh_dict = kwargs["mh_dict"]
        self.scales = []

        # self.results = dict()

        for p in self.processes:
            p.start()

    def add_to_queue(self, item):
        self.queue.put(item)

    def dump_all_injection_values(self):
        for scale in self.scales:
            self.mh.dump_injection_values(scale)

    def run_trial(self, **kwargs):

        mh_dict = kwargs["mh_dict"]

        mpmh = generate_dynamic_mh_class(mh_dict)

        while True:
            item = self.queue.get()
            if item is None:
                break

            (scale, seed) = item

            full_dataset = self.mh.prepare_dataset(scale, seed)

            mpmh.run_single(full_dataset, scale, seed)
            self.queue.task_done()

    def fill_queue(self):
        scale_range, n_trials = self.mh.trial_params(self.mh_dict)

        self.scales = scale_range

        for scale in scale_range:
            for _ in range(n_trials):
                r.add_to_queue((scale, int(random.random() * 10 ** 8)))

    def terminate(self):
        """ wait until queue is empty and terminate processes """
        self.queue.join()
        for p in self.processes:
            p.terminate()

        self.dump_all_injection_values()


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path for analysis pkl_file")
    parser.add_argument("-n", "--n_cpu", default=os.cpu_count()-1)
    cfg = parser.parse_args()

    logging.info("N CPU available {0}".format(os.cpu_count()))

    with open(cfg.file, "rb") as f:
        mh_dict = pickle.load(f)

    r = MultiProcessor(n_cpu=cfg.n_cpu, mh_dict=mh_dict)
    r.fill_queue()
    r.terminate()

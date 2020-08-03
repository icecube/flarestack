import os
from flarestack.shared import host_server, make_analysis_pickle
from flarestack.cluster.run_desy_cluster import submit_to_cluster,\
    wait_for_cluster
from flarestack.cluster.make_desy_cluster_script import make_desy_submit_file
from flarestack.core.multiprocess_wrapper import run_multiprocess

if host_server == "DESY":
    submit_cluster = submit_to_cluster
    wait_cluster = wait_for_cluster

else:
    def submit_cluster(path, **kwargs):
        raise Exception("No cluster submission script recognised!")

    def wait_cluster(*args, **kwargs):
        raise Exception("No cluster waiting script recognised!")


def submit_local(mh_dict, n_cpu):
    run_multiprocess(n_cpu=n_cpu, mh_dict=mh_dict)


def analyse(mh_dict, cluster=False, n_cpu=None, **kwargs):
    """Generic function to run an analysis on a given MinimisationHandler
    dictionary. Can either run on cluster, or locally, based on the boolean
    cluster arg. The number of cpus can be specified, as well as specific
    kwargs such as number of jobs to run.

    :param mh_dict: MinimisationHandler dictionary
    :param cluster: Boolean flag for whether to run on cluster or locally.
    Default is False (i.e locally)
    :param n_cpu: Number of CPUs to run with. Should be 1 for submit to cluster
    :param kwargs: Optional kwargs
    """

    path = make_analysis_pickle(mh_dict)

    job_id = None

    if cluster:

        if n_cpu is None:
            n_cpu = 1

        job_id = submit_cluster(path, n_cpu=n_cpu, **kwargs)
    else:
        if n_cpu is None:
            n_cpu = min(os.cpu_count() - 1, 32)
        else:
            n_cpu = min(n_cpu, os.cpu_count() -1)
        submit_local(mh_dict, n_cpu=n_cpu)

    return job_id

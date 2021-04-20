import logging
from flarestack.cluster.submitter import Submitter

logger = logging.getLogger(__name__)


def submit_local(mh_dict, n_cpu):
    s = Submitter.submitter_dict['local'](
        mh_dict, use_cluster=False, n_cpu=n_cpu, do_sensitivity_scale_estimation=False
    )
    s.analyse()


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
    if cluster and ("n_jobs" in kwargs):
        logger.warning('The Submitter class determines the number of jobs '
                       'from the number of trials and the number of trials per job. '
                       'So it is not necessary anymore to specifically give the number of jobs!')
        ntrials_orig = mh_dict['n_trials']
        mh_dict['n_trials'] = ntrials_orig * kwargs['n_jobs']
        kwargs['trials_per_task'] = ntrials_orig

    s = Submitter.get_submitter(mh_dict, cluster, n_cpu, **kwargs)
    s.analyse()
    return s.job_id


def wait_cluster(job_ids=None):
    Submitter.wait_for_cluster(job_ids)

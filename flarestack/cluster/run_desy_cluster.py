"""Script to run stacking scripts on the DESY cluster.

Through use of argparse, a given configuration for the code can be selected.
This can be given from the command line, in the form:

python RunCluster.py -c Desired_Configuration_Name -n Number_Of_Tasks -s

Each available configuration must be listed in "config.ini", and controls
options for fitting, such as which catalogue is to be used, and which seasons
of data should be included. If -x is included, then a new job is submitted
to the cluster. Having submitted the job to the cluster it will be run in
parallel Number_of_Tasks times. The shell script SubmitOne.sh is called for
each task, which in turn calls RunLocal.py with the given configuration setting.

The Wait function will periodically query the cluster
to check on the status of the job, and will output the job status occasionally.

Once all sub-tasks are completed, the script will proceed to call
MergeFiles.run() for the given configuration, combining results.

"""
import subprocess
import time
import os
import os.path
import logging
import argparse
import numpy as np
from flarestack.shared import log_dir, fs_dir
from flarestack.cluster.submitter import Submitter
from flarestack.cluster.make_desy_cluster_script import make_desy_submit_file, submit_file

logger = logging.getLogger(__name__)

username = os.path.basename(os.environ['HOME'])

cmd = 'qstat -u ' + username


def wait_for_cluster(job_ids=None):
    logger.warning('The wait_for_cluster function is deprecated! '
                   'Use the Submitter class instead.')
    Submitter.wait_for_cluster(job_ids)

    # if not job_ids:
    #     wait_for_job()
    # else:
    #     try:
    #         for i, job_id in enumerate(job_ids):
    #
    #             logger.debug(f'waiting for job {job_id}')
    #             prog_str = f'{i}/{len(job_ids)}'
    #             wait_for_job(job_id, prog_str)
    #
    #     except TypeError:
    #         logger.debug('Only waiting for one job')
    #         wait_for_job(job_ids)


def wait_for_job(job_id=None, progress_str=None):
    """
    Runs the command cmd, which queries the status of the job on the
    cluster, and reads the output. While the output is not an empty
    string (indicating job completion), the cluster is re-queried
    every 30 seconds. Occasionally outputs the number of remaining sub-tasks
    on cluster, and outputs full table result every ~ 8 minutes. On
    completion of job, terminates function process and allows the script to
    continue.
    """

    if not job_id:
        job_id_str = 's'
    else:
        if progress_str:
            job_id_str = f' {progress_str} {job_id}'
        else:
            job_id_str = ' ' + str(job_id)

    time.sleep(10)

    cmd = f'qstat -u {username}'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    tmp = process.stdout.read().decode()
    n_total = n_tasks(tmp, job_id)
    i = 31
    j = 6
    while n_total != 0:
        if i > 3:

            running_process = subprocess.Popen(
                cmd + " -s r", stdout=subprocess.PIPE, shell=True)
            running_tmp = running_process.stdout.read().decode()

            if running_tmp != '':
                n_running = n_tasks(running_tmp, job_id)
            else:
                n_running = 0

            logger.info(f'{time.asctime(time.localtime())} - Job{job_id_str}:'
                         f' {n_total} entries in queue. '
                         f'Of these, {n_running} are running tasks, and '
                         f'{n_total-n_running} are tasks still waiting to be executed.')
            i = 0
            j += 1

        if j > 7:
            logger.info(str(tmp))
            j = 0

        time.sleep(30)
        i += 1
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        tmp = process.stdout.read().decode()
        n_total = n_tasks(tmp, job_id)


def submit_to_cluster(path, n_cpu=2, n_jobs=10, ram_per_core=None, **kwargs):

    for file in os.listdir(log_dir):
        os.remove(log_dir + file)

    # Submits job to the cluster

    submit_cmd = "qsub "

    if n_cpu > 1:
        submit_cmd += " -pe multicore {0} -R y ".format(n_cpu)

    ram_per_core = "{0:.1f}G".format(6./float(n_cpu) + 2.) if not ram_per_core else ram_per_core
    print("Ram per core:", ram_per_core)

    submit_cmd += "-t 1-{0}:1 {1} {2} {3}".format(
        n_jobs, submit_file, path, n_cpu
    )

    make_desy_submit_file(ram_per_core, **kwargs)

    print(time.asctime(time.localtime()), submit_cmd, "\n")

    process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
    msg = process.stdout.read().decode()
    print(msg)
    job_id = int(str(msg).split('job-array')[1].split('.')[0])

    return job_id


def n_tasks(tmp, job_id):
    """
    Returns the number of tasks given the output of qsub
    :param tmp: output of qsub
    :param job_id: int, optional, if given only tasks belonging to this job will we counted
    :return: int
    """
    st = str(tmp)
    ids = np.array([int(s.split(' ')[2]) for s in st.split('\n')[2:-1]])

    if job_id:
        return len(ids[ids == job_id])
    else:
        return len(ids)


if not os.path.isfile(submit_file):
    make_desy_submit_file()

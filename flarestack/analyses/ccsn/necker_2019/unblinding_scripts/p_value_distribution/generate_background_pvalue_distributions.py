import logging, os, pickle, subprocess, time
from multiprocessing import Pool
from tqdm import tqdm

from flarestack.cluster.submitter import DESYSubmitter
from flarestack.shared import cluster_dir, log_dir, fs_scratch_dir
from flarestack.analyses.ccsn.necker_2019.unblinding_scripts.p_value_distribution.generate_background_pvalue_distribution_single import (
    single_filename,
    run_background_pvalue_trials,
    p_value_filename,
    p_value_directory,
)


logging.getLogger().setLevel("DEBUG")
logging.debug("logging level is DEBUG")
logging.getLogger("matplotlib").setLevel("INFO")
logger = logging.getLogger("main")


# -------------------------------------------------------------------------------------
# START Submitter subclass
# -------------------------------------------------------------------------------------


class MyPValueSubmitter(DESYSubmitter):

    """
    This is a bit hacky. We will subclass the Submitter class and put a different submission script in,
    that no longer runs the MultiProcessor on the cluster but our run_background_pvalue_trials() function.
    NOTE: So far this is only implemented to run on the DESY cluster. If you are on a different cluster,
          you can only run trials locally (set use_cluster to False)
    """

    submit_file = os.path.join(cluster_dir, "MyPValueSubmitterSubmitScript.sh")

    def __init__(self, param_dict, use_cluster, **cluster_kwargs):
        super().__init__(param_dict, use_cluster, **cluster_kwargs)
        self.do_sensitivity_scale_estimation = False

    def submit_cluster(self, param_dict):
        """Submits the job to the cluster"""
        # if specified, remove old logs from log directory
        if self.remove_old_logs:
            self.clear_log_dir()

        # make sure the result directory exists
        if not os.path.isdir(p_value_directory):
            logger.debug(f"making directory at {p_value_directory}")
            os.makedirs(p_value_directory)

        # Get the number of tasks that will have to be submitted in order to get ntrials
        ntrials = param_dict["n_trials"]
        n_tasks = int(ntrials / self.trials_per_task)
        logger.debug(f"running {ntrials} trials in {n_tasks} tasks")

        # assemble the submit command
        submit_cmd = MyPValueSubmitter.submit_cmd
        if self.cluster_cpu > 1:
            submit_cmd += " -pe multicore {0} -R y ".format(self.cluster_cpu)
        submit_cmd += (
            f"-t 1-{n_tasks}:1 {MyPValueSubmitter.submit_file} {self.trials_per_task}"
        )
        logger.debug(f"Ram per core: {self.ram_per_core}")
        logger.info(f"{time.asctime(time.localtime())}: {submit_cmd}")

        self.make_cluster_submission_script()

        process = subprocess.Popen(submit_cmd, stdout=subprocess.PIPE, shell=True)
        msg = process.stdout.read().decode()
        logger.info(str(msg))
        self.job_id = int(str(msg).split("job-array")[1].split(".")[0])

    def submit_local(self, param_dict):
        ntrials = param_dict["n_trials"]
        itr = list(range(ntrials))

        if self.n_cpu > 1:
            with Pool(self.n_cpu) as p:
                multi_res = p.map(run_background_pvalue_trials, itr)
                p.close()
                p.join()

            res = self.combine_results_dictionaries(multi_res)

        else:
            res = run_background_pvalue_trials(666, ntrials)

        with open(p_value_filename, "wb") as f:
            pickle.dump(res, f)

    def make_cluster_submission_script(self):
        """Produces the shell script used to run on the DESY cluster."""
        flarestack_scratch_dir = os.path.dirname(fs_scratch_dir[:-1]) + "/"

        text = (
            "#!/bin/zsh \n"
            "## \n"
            "##(otherwise the default shell would be used) \n"
            "#$ -S /bin/zsh \n"
            "## \n"
            "##(the running time for this job) \n"
            f"#$ -l h_cpu={self.h_cpu} \n"
            "#$ -l h_rss=" + str(self.ram_per_core) + "\n"
            "## \n"
            "## \n"
            "##(send mail on job's abort) \n"
            "#$ -m a \n"
            "## \n"
            "##(stderr and stdout are merged together to stdout) \n"
            "#$ -j y \n"
            "## \n"
            "## name of the job \n"
            "## -N MyPValueDistribution " + MyPValueSubmitter.username + " \n"
            "## \n"
            "##(redirect output to:) \n"
            "#$ -o /dev/null \n"
            "## \n"
            "sleep $(( ( RANDOM % 60 )  + 1 )) \n"
            'exec > "$TMPDIR"/${JOB_ID}_stdout.txt '
            '2>"$TMPDIR"/${JOB_ID}_stderr.txt \n'
            "eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh) \n"
            "export PYTHONPATH=" + MyPValueSubmitter.root_dir + "/ \n"
            "export FLARESTACK_SCRATCH_DIR=" + flarestack_scratch_dir + " \n"
            "python " + single_filename + " --ntrials $1\n"
            "cp $TMPDIR/${JOB_ID}_stdout.txt " + log_dir + "\n"
            "cp $TMPDIR/${JOB_ID}_stderr.txt " + log_dir + "\n "
        )

        logger.info("Creating file at {0}".format(MyPValueSubmitter.submit_file))

        with open(MyPValueSubmitter.submit_file, "w") as f:
            f.write(text)

        logger.debug("Bash file created: \n {0}".format(text))

        cmd = "chmod +x " + MyPValueSubmitter.submit_file
        os.system(cmd)

    def _clean_injection_values_and_pickled_results(self):
        pass

    @staticmethod
    def combine_results_dictionaries(result_dictionaries):
        res = result_dictionaries[0]

        for r in tqdm(result_dictionaries[1:], desc="collecting results"):
            for pdf_type, full_res in r.items():
                for cat, cat_res in full_res.items():
                    for time_key, time_res in cat_res.items():
                        for gamma, p_values in time_res.items():
                            res[pdf_type][cat][time_key][gamma].extend(p_values)

        return res

    def collect_cluster_results(self):
        files = os.listdir(p_value_directory)
        result_dictionaries = list()

        for fi in files:
            fn = os.path.join(p_value_directory, fi)

            if os.path.isfile(fn) and fn.endswith(".pkl"):
                with open(fn, "rb") as f:
                    result_dictionaries.append(pickle.load(f))
            else:
                logger.warning(f"Could not load {fi}: Not a file!")

        res = self.combine_results_dictionaries(result_dictionaries)

        if os.path.isfile(p_value_filename):
            ipt = input(
                f"File {p_value_filename} already exists. \n"
                f"Continue and overwrite? [y/n] "
            )
            if ipt == "y":
                pass
            else:
                logger.warning("Not writing results to disk!")
                return

        logger.info(f"Writing results to {p_value_filename}")
        with open(p_value_filename, "wb") as f:
            pickle.dump(res, f)


# -------------------------------------------------------------------------------------
# END Submitter subclass
# -------------------------------------------------------------------------------------


if __name__ == "__main__":
    p_dict = {
        "n_trials": 1000,
    }

    cluster_kwargs = {"cluster_cpu": 1, "trials_per_task": 10, "h_cpu": "02:59:59"}

    n_cpu = 30
    use_cluster = (
        True  # NOTE: So far this is only implemented to run on the DESY cluster.
    )
    #                           If you are on a different cluster, you can only run trials locally
    #                           (set use_cluster to False)

    sub = MyPValueSubmitter(
        param_dict=p_dict, use_cluster=use_cluster, n_cpu=n_cpu, **cluster_kwargs
    )
    sub.analyse()

    if use_cluster:
        sub.wait_for_job()
        sub.collect_cluster_results()

import logging
import os
from flarestack.shared import fs_dir, log_dir, fs_scratch_dir

logger = logging.getLogger(__name__)

username = os.path.basename(os.environ['HOME'])

root_dir = os.path.dirname(fs_dir[:-1])

cluster_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

submit_file = cluster_dir + "SubmitDESY.sh"

flarestack_scratch_dir = os.path.dirname(fs_scratch_dir[:-1]) + "/"


def make_desy_submit_file(ram_per_core="6.0G", h_cpu='23:59:00'):

    text = "#!/bin/zsh \n" \
           "## \n" \
           "##(otherwise the default shell would be used) \n" \
           "#$ -S /bin/zsh \n" \
           "## \n" \
           "##(the running time for this job) \n" \
          f"#$ -l h_cpu={h_cpu} \n" \
           "#$ -l h_rss=" + str(ram_per_core) + "\n" \
           "## \n" \
           "## \n" \
           "##(send mail on job's abort) \n" \
           "#$ -m a \n" \
           "## \n" \
           "##(stderr and stdout are merged together to stdout) \n" \
           "#$ -j y \n" \
           "## \n" \
           "## name of the job \n" \
           "## -N Flarestack script " + username + " \n" \
           "## \n" \
           "##(redirect output to:) \n" \
           "#$ -o /dev/null \n" \
           "## \n" \
           "sleep $(( ( RANDOM % 60 )  + 1 )) \n" \
           'exec > "$TMPDIR"/${JOB_ID}_stdout.txt ' \
           '2>"$TMPDIR"/${JOB_ID}_stderr.txt \n' \
           'eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh) \n' \
           'export PYTHONPATH=' + root_dir + '/ \n' \
           'export FLARESTACK_SCRATCH_DIR=' + flarestack_scratch_dir + " \n" \
           'python ' + fs_dir + 'core/multiprocess_wrapper.py -f $1 -n $2 \n' \
           'cp $TMPDIR/${JOB_ID}_stdout.txt ' + log_dir + '\n'\
           'cp $TMPDIR/${JOB_ID}_stderr.txt ' + log_dir + '\n '

    logger.info("Creating file at {0}".format(submit_file))

    with open(submit_file, "w") as f:
        f.write(text)

    logger.debug("Bash file created: \n {0}".format(text))

    cmd = "chmod +x " + submit_file

    os.system(cmd)


if __name__ == "__main__":
    make_desy_submit_file()

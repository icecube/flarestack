import logging
import os

from flarestack.shared import fs_dir, log_dir

username = os.path.basename(os.environ['HOME'])

root_dir = os.path.dirname(fs_dir[:-1])

cluster_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

local_submit_file = cluster_dir + "SubmitLocal.sh"


def make_local_submit_file():

    text = "#!/bin/zsh \n" \
           "## \n" \
           'export PYTHONPATH=`which python`:' + root_dir + "/ \n" \
           'python ' + fs_dir + 'core/multiprocess_wrapper.py -f $1 -n $2'

    logging.info("Creating file at {0}".format(local_submit_file))

    with open(local_submit_file, "w") as f:
        f.write(text)

    logging.info("Bash file created:")
    logging.info(text)

    cmd = "chmod +x " + local_submit_file

    os.system(cmd)

    logging.info("CMD: {0}".format(cmd))


if __name__ == "__main__":
    make_local_submit_file()

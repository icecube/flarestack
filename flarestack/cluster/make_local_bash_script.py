from __future__ import print_function
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

    print("Creating file at", local_submit_file)

    with open(local_submit_file, "w") as f:
        f.write(text)

    print("Bash file created: \n")
    print(text)

    cmd = "chmod +x " + local_submit_file

    os.system(cmd)

    print("CMD:", cmd)


if __name__ == "__main__":
    make_local_submit_file()

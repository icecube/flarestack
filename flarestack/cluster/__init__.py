import time
import os
from flarestack.shared import host_server, fs_dir
from flarestack.cluster.run_desy_cluster import submit_to_cluster
from flarestack.cluster.make_local_bash_script import local_submit_file,\
    make_local_submit_file

if host_server == "DESY":
    submit_cluster = submit_to_cluster

else:
    def submit_cluster(path, **kwargs):
        raise Exception("No cluster submission script recognised!")


if not os.path.isfile(local_submit_file):
    make_local_submit_file()


def submit_local(path):

    bashfile = local_submit_file

    submit_cmd = bashfile + " " + path

    print(time.asctime(time.localtime()), submit_cmd, "\n")
    os.system(submit_cmd)


def submit(path, cluster=False, **kwargs):
    if cluster:
        submit_cluster(path, **kwargs)
    else:
        submit_local(path)


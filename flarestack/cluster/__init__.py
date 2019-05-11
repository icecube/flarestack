import time
import os
from flarestack.shared import host_server, fs_dir
from flarestack.cluster.run_desy_cluster import submit_to_cluster

if host_server == "DESY":
    submit = submit_to_cluster

else:
    def submit(path, **kwargs):
        raise Exception("No cluster submission script recognised!")


def submit_local(path, bashname="SubmitDESY.sh"):

    bashfile = fs_dir + "cluster/" + bashname

    submit_cmd = bashfile + " " + path

    print(time.asctime(time.localtime()), submit_cmd, "\n")
    os.system(submit_cmd)

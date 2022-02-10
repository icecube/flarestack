import os


sl_data_dir_raw = os.path.dirname(os.path.realpath(__file__))


def sl_data_dir(sindec=None):
    if not sindec:
        return sl_data_dir_raw
    else:
        return sl_data_dir_raw + "/{:.4f}".format(sindec)


def sl_ps_data():
    return sl_data_dir_raw + "/ps.npy"

import os
import numpy as np
from flarestack.utils.prepare_catalogue import single_source


home_dir = os.environ['HOME']
output_dir_for_sl = home_dir + '/skylab/skylab/fs_crosscheck/data/'
output_dir_for_fs = os.path.dirname(os.path.realpath(__file__)) + '/data/'

np.random.seed(1)
Nsources = 500

sindecs = np.random.uniform(-1, 1, Nsources)

sources = single_source(sindecs[0])
nsources = [1, 5, 10, 100, 200, 500]

for sindec in sindecs[1:]:
    j = len(sources)
    sources = np.append(sources, single_source(sindec))
    if len(sources) != j+1:
        raise Exception('j={0} and len(sources)={1}'.format(j, len(sources)))

for dir in [output_dir_for_fs, output_dir_for_sl]:
    np.save(dir + 'sources.npy', sources)

for n in nsources:
    np.save(output_dir_for_fs + str(n) + 'sources.npy', sources[:n])


def fs_sources(i):
    return output_dir_for_fs + str(nsources[i]) + 'sources.npy'


def sl_sources(i):
    return sources[:nsources[i]]

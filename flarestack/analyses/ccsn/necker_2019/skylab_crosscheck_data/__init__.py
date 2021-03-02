import os
import numpy as np
import logging
from pathlib import Path

from flarestack.shared import host_server


logger = logging.getLogger(__name__)
# dir_path = os.path.dirname(os.path.realpath(__file__))
if host_server == 'DESY':
    dir_path = '/afs/ifh.de/user/n/neckerja/scratch/skylab_scratch/skylab_output/data/sn_catalogue_crosscheck'
elif host_server == 'WIPAC':
    dir_path = '/data/user/jnecker/skylab_output/data/sn_catalogue_crosscheck'
else:
    raise NotImplementedError


def skylab_data_file_path(gamma, cat, northern=False, weighted=False, ps=False, seasons=None, kernel=None,
                          spatial_only=False):
    nstr = '_northern' if northern else ''
    wstr = '_weighted' if weighted else ''
    ps_str = '_ps' if ps else ''
    season_str = ''
    kernel_str = '' if isinstance(kernel, type(None)) else f'_kernel{kernel}'
    spatial_str = '_spatial_only' if spatial_only else ''
    if seasons:
        for s in seasons:
            season_str += f'_IC86_1' if s == 'IC86_2011' else f'_{s}'
        season_str += '_'
    return f'{dir_path}{season_str}/{cat}_{gamma:.2f}{nstr}{wstr}{ps_str}{kernel_str}{season_str}{spatial_str}.npy'


def skylab_data(*args, **kwargs):
    fn = skylab_data_file_path(*args, **kwargs)
    if os.path.isfile(fn):
        logger.debug(f'loading {fn}')
        return np.load(fn)
    else:
        logger.debug(f'No file {fn}')
        return None

def skylab_northern_only_data(cat, gamma, kernel=0, seasons=None):
    fn = f'{Path.home()}/scratch/skylab_scratch/skylab_output/data/sn_catalogue_crosscheck_northern/' \
         f'{cat}_{gamma:.2f}_kernel{kernel}_northern.npy'
    if os.path.isfile(fn):
        logger.debug(f'loading {fn}')
        return np.load(fn)
    else:
        logger.debug(f'No file {fn}')
        return None


def skylab_spatial_only_data(cat, gamma, kernel=0, season=None):
    season_str = '' if not season else '_' + season
    fn = f'{Path.home()}/scratch/skylab_scratch/skylab_output/data/sn_catalogue_crosscheck{season_str}/' \
         f'{cat}_{gamma:.2f}_kernel{kernel}_spatial_only{season_str}.npy'
    if os.path.isfile(fn):
        logger.debug(f'loading {fn}')
        return np.load(fn)
    else:
        logger.debug(f'No file {fn}')
        return None

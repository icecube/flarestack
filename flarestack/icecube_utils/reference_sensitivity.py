import os
import numpy as np
from scipy.interpolate import interp1d, interp2d
from flarestack.data.icecube.ic_season import get_published_sens_ref_dir

published_sens_ref_dir = get_published_sens_ref_dir()

def reference_sensitivity(sindec=np.array(0.0), gamma=2.0):
    """Interpolates between the saved values of the Stefan Coenders 7 year PS
    analysis sensitivity. Then converts given values for sin(declination to
    the equivalent reference sensitivity. Adds values for Sindec = +/- 1,
    equal to nearest known value.

    :param sindec: Sin(declination)
    :return: 7 year PS sensitivity at sindec
    """
    skylab_sens_path = published_sens_ref_dir + "sens.npy"
    data = np.load(skylab_sens_path)
    sindecs = np.sin(np.array([x[0] for x in data]))
    gammas = [1.0, 2.0, 3.0]

    # The sensitivities here are given in units TeV ^ -gamma per cm2 per s
    # The sensitivities used in this code are GeV ^-1 per cm2 per s
    # The conversion is thus (TeV/Gev) ^ (1 - gamma) , i.e 10 ** 3(1-gamma)
    sens = np.array([list(x)[1:] for x in data]) * 10 ** 3
    scaling = np.array([10 ** (3 * (i - 1)) for i in range(3)])
    sens *= scaling

    # Extend range of sensitivity to +/- 1 through approximation,

    sindecs = np.append(-1, sindecs)
    sindecs = np.append(sindecs, 1)

    sens = np.vstack((sens[0], sens))
    sens = np.vstack((sens, sens[-1]))
    sens_ref = interp2d(np.array(sindecs), np.array(gammas), np.log(sens.T))

    if np.array(sindec).ndim > 0:
        return np.array([np.exp(sens_ref(x, gamma))[0] for x in sindec])
    else:
        return np.exp(sens_ref(sindec, gamma))


def reference_7year_discovery_potential(sindec=0.0, gamma=2.0):
    """Interpolates between the saved values of the Stefan Coenders 7 year PS
    analysis discovery potential. Then converts given values for sin(
    declination to the equivalent reference sensitivity. Adds values for Sindec
    = +/- 1, equal to nearest known value.

    :param sindec: Sin(declination)
    :return: 7 year PS discovery potential at sindec
    """
    skylab_disc_path = published_sens_ref_dir + "disc.npy"
    data = np.load(skylab_disc_path)
    sindecs = np.sin(np.array([x[0] for x in data]))
    gammas = [1.0, 2.0, 3.0]

    # The discovery potentials are given in units TeV ^ -gamma per cm2 per s
    # The discovery potentials used in this code are GeV ^-1 per cm2 per s
    # The conversion is thus (TeV/Gev) ^ (1 - gamma) , i.e 10 ** 3(1-gamma)
    disc = np.array([list(x)[1:] for x in data]) * 10 ** 3
    scaling = np.array([10 ** (3 * (i - 1)) for i in range(3)])
    disc *= scaling

    # Extend range of sensitivity to +/- 1 through approximation,

    sindecs = np.append(-1, sindecs)
    sindecs = np.append(sindecs, 1)

    disc = np.vstack((disc[0], disc))
    disc = np.vstack((disc, disc[-1]))
    disc_ref = interp2d(np.array(sindecs), np.array(gammas), np.log(disc.T))

    if np.array(sindec).ndim > 0:
        return np.array([np.exp(disc_ref(x, gamma))[0] for x in sindec])
    else:
        return np.exp(disc_ref(sindec, gamma))

import logging
import matplotlib.pyplot as plt
import numpy as np

# Set Logger Level
logging.getLogger().setLevel("DEBUG")
logging.getLogger('matplotlib').setLevel('INFO')


def plot_dec_sens(dec, sens, fname):

    if len(dec) != len(sens):
        logging.debug(sens)
        logging.debug(dec)
        raise Exception

    # convert sensitivity to TeV /cm2 /s
    sens = np.array(sens) * 1e-3

    sin_dec = np.sin(dec)

    fig, ax = plt.subplots()

    ax.plot(sin_dec, sens, 'o', label='this analysis')
    ax.set_ylabel('$E^2 \frac{dN}{dE}$ in $TeV \, cm^{-2} \, s^{-1}$')
    ax.set_xlabel('sin($\delta$)')
    ax.set_yscale('log')
    ax.legend()

    plt.grid()
    plt.title('sample sensitivity')
    plt.tight_layout()

    plt.savefig(fname)


sn_colors = {'IIp': 'gold', 'IIn': 'red', 'Ibc': 'blue'}


def get_sn_color(type):
    color = sn_colors.get(type, None)
    if not color and type == 'IIP':
        sn_colors.get('IIp', None)
    if not color:
        raise Exception('SN type {0} not in dictionary!'.format(type))
    return color

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


sn_colors = {'IIp': 'orange', 'IIn': 'red', 'Ibc': 'blue'}


def get_sn_color(type):
    if type == 'IIP':
        return sn_colors['IIp']
    else:
        return sn_colors[type]

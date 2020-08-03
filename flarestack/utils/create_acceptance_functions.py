import numpy as np
import os
import pickle
import logging
from flarestack.shared import acceptance_path, get_base_sob_plot_dir
from flarestack.core.energy_pdf import PowerLaw
from flarestack.utils.make_SoB_splines import make_plot

logger = logging.getLogger(__name__)

sin_dec_range = np.linspace(-1, 1, 101)
sin_edges = np.append(-1., (sin_dec_range[1:] + sin_dec_range[:-1])/ 2.)
sin_edges = np.append(sin_edges, 1.)
dec_range = np.arcsin(sin_dec_range)
dec_edges = np.arcsin(sin_edges)

gamma_vals = np.linspace(0.5, 5.5, 201)


def make_acceptance_f(all_data):

    for season in all_data.values():
        acc_path = acceptance_path(season)
        make_acceptance_season(season, acc_path)


def make_acceptance_season(season, acc_path):
    e_pdf = PowerLaw()

    mc = season.get_pseudo_mc()

    acc = np.ones((len(dec_range), len(gamma_vals)), dtype=np.float)

    for i, dec in enumerate(dec_range):

        # Sets half width of band
        dec_width = np.deg2rad(5.)

        # Sets a declination band 5 degrees above and below the source
        min_dec = max(-np.pi / 2., dec - dec_width)
        max_dec = min(np.pi / 2., dec + dec_width)
        # Gives the solid angle coverage of the sky for the band
        omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

        band_mask = np.logical_and(np.greater(mc["trueDec"], min_dec),
                                   np.less(mc["trueDec"], max_dec))

        cut_mc = mc[band_mask]

        for j, gamma in enumerate(gamma_vals):
            weights = e_pdf.weight_mc(cut_mc, gamma)
            acc[i][j] = np.sum(weights / omega)

    try:
        os.makedirs(os.path.dirname(acc_path))
    except OSError:
        pass

    logger.info("Saving {0} acceptance values to: {1}".format(season.season_name, acc_path))

    with open(acc_path, "wb") as f:
        pickle.dump([dec_range, gamma_vals, acc], f)

    del mc

    savepath = get_base_sob_plot_dir(season) + "acceptance_f.pdf"

    try:
        os.makedirs(os.path.dirname(savepath))
    except OSError:
        pass

    make_plot(acc, savepath, gamma_vals, np.sin(dec_range), label_x=r"$\gamma$", label_y=r"$\sin(\delta)$")

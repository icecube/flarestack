import numpy as np
import os
import cPickle as Pickle
from flarestack.shared import gamma_range, acceptance_path
from flarestack.core.energy_PDFs import PowerLaw
from flarestack.core.injector import Injector
from flarestack.utils.dataset_loader import data_loader


sin_dec_range = np.linspace(-1, 1, 101)
sin_edges = np.append(-1., (sin_dec_range[1:] + sin_dec_range[:-1])/ 2.)
sin_edges = np.append(sin_edges, 1.)
dec_range = np.arcsin(sin_dec_range)
dec_edges = np.arcsin(sin_edges)

# gamma_vals = np.linspace(gamma_range[0], gamma_range[1], 31)
gamma_vals = np.linspace(0.5, 5.5, 201)

def make_acceptance_f(all_data):
    e_pdf = PowerLaw()

    for season in all_data:
        try:
            mc = data_loader(season["mc_path"])
            # old_dec_bins = np.load(
            #     season['aw_path'] + '_bins_dec.npy')
            # old_gamma_bins = np.load(
            #     season['aw_path'] + '_bins_gamma.npy')
            #
            # old_values = np.load(season['aw_path'] + '_values.npy')

            acc = np.ones((len(dec_range), len(gamma_vals)), dtype=np.float)

            for i, dec in enumerate(dec_range):

                # Sets half width of band
                dec_width = np.deg2rad(5.)

                # Sets a declination band 5 degrees above and below the source
                min_dec = max(-np.pi / 2., dec - dec_width)
                max_dec = min(np.pi / 2., dec + dec_width)
                # Gives the solid angle coverage of the sky for the band
                omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

                # # Sets a declination band 5 degrees above and below the source
                # min_dec = dec_edges[i]
                # max_dec = dec_edges[i+1]
                #
                # # Gives the solid angle coverage of the sky for the band
                # omega = 2. * np.pi * (np.sin(max_dec) - np.sin(min_dec))

                band_mask = np.logical_and(np.greater(mc["trueDec"], min_dec),
                                               np.less(mc["trueDec"], max_dec))

                cut_mc = mc[band_mask]

                for j, gamma in enumerate(gamma_vals):
                    weights = e_pdf.weight_mc(cut_mc, gamma)
                    acc[i][j] = np.sum(weights / omega)

            acc_dict = {
                "dec": dec_range,
                "gamma": gamma_vals,
                "acceptance": acc
            }

            savepath = acceptance_path(season)

            try:
                os.makedirs(os.path.dirname(savepath))
            except OSError:
                pass

            print "Saving", season["Name"], "acceptance values to:", savepath

            with open(savepath, "wb") as f:
                Pickle.dump(acc_dict, f)

        except IOError:
            pass

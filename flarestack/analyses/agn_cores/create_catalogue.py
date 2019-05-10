"""Script to reproduce the catalogues used in Alexander Stasik's CCSN
stacking analysis. Raw files were provided by him, following the merger of
several supernova catalogues.

The analysis created two subcatalogues for each class. One, a nearby sample,
contained 70% of the signal weight. Another larger sample contained the
remaining 30% weight distributed across many sources. As integer numbers of
sources are used, the closest percentage to 70% is used for splitting.
"""
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from flarestack.analyses.agn_cores.shared_agncores import raw_cat_dir,\
    agn_catalogue_name, agn_cores_output_dir
from flarestack.analyses.agn_cores.shared_agncores import create_random_src, \
    plot_catalogue
from flarestack.utils.prepare_catalogue import cat_dtype
import astropy.io.fits as pyfits
from astropy.table import Table
import numpy as np
import os

def select_nrandom_sources(cat, n_random=100):
    # select n_random random sources

    # import random
    # list_of_random_srcs = random.sample(raw_cat, 100)
    # raw_cat = list_of_random_srcs
    import pandas as pd

    df = cat.to_pandas()
    df_random = df.sample(n=n_random)
    cat_new = Table.from_pandas(df_random)
    print(cat_new)
    return cat_new


'''Open original (complete) catalogue'''
raw_cat = pyfits.open(raw_cat_dir+'radioloud_rxs_allwise_nvss_no3LACbll.fits')
raw_cat = Table(raw_cat[1].data, masked=True)

raw_cat = raw_cat[raw_cat['DEC']>0]   # Select Northen sky sources only
raw_cat = raw_cat.group_by('2RXS_SRC_FLUX')  # order catalog by flux
#####################################
#    Select 100 brightest sources   #
#####################################
raw_cat= raw_cat[-100:]

#####################################
#      Select two close sources     #
#####################################
raw_cat = raw_cat[(raw_cat['RA_DEG']>220)&(raw_cat['RA_DEG']<240)&(raw_cat['DEC_DEG']>7.1)&(raw_cat['DEC_DEG']<15)]
plot_catalogue(raw_cat["RA_DEG"], raw_cat["DEC_DEG"], src_weight= np.ones(len(raw_cat)),
               filename = 'NorthSky_close_sources',
               plot_path = agn_cores_output_dir('catalogues'))


# raw_weight = []
# for w in raw_cat['2RXS_SRC_FLUX']:
#     raw_weight.append(w)

#####################################
#      Create 100 random sources    #
#####################################
# random_ra, random_dec = create_random_src(min_distance=10, nr_sources=100) # in degree
# plot_catalogue(random_ra, random_dec, src_weight= np.ones(len(raw_cat)),
#                filename = 'randomSrc_100brightest_NorthSky_equatorial_weight1',
#                plot_path = agn_cores_output_dir('catalogues'))
#
new_cat = np.empty(len(raw_cat), dtype=cat_dtype)
new_cat["ra"] = np.deg2rad(raw_cat["RA_DEG"])  # NVSS RA in radians
new_cat["dec"] = np.deg2rad(raw_cat["DEC_DEG"]) # NVSS DEC in radians

# new_cat["ra"] = np.deg2rad(random_ra)
# new_cat["dec"] = np.deg2rad(random_dec)
new_cat["Distance (Mpc)"] = np.ones(len(raw_cat))
new_cat["Ref Time (MJD)"] = np.ones(len(raw_cat))
# new_cat["Relative Injection Weight"] = raw_cat["2RXS_SRC_FLUX"]*1e13
new_cat["Relative Injection Weight"] = np.ones(len(raw_cat))  # set equal weights

# save name of source (if given)
src_name = []
for vv10, rxs in zip(raw_cat['NAME_vv10'], raw_cat['2RXS_ID']):
    if (vv10!='N/A'):
        src_name.append(vv10)
    else:
        src_name.append(rxs)

new_cat["Name"] = src_name

# save_path = agn_catalogue_name("radioloud", "2rxs_100brightest_srcs")
# save_path = agn_catalogue_name("random", "NorthSky_100brightest_srcs_dec0_weight1")
save_path = agn_catalogue_name("random", "NorthSky_2close_srcs")

print("Saving to", save_path)
np.save(save_path, new_cat)
#
#
# # plot_catalogue(raw_cat["RA_DEG"], raw_cat["DEC_DEG"], src_weight= np.ones(len(raw_cat)),
# #                filename = 'NorthSky_100brightest_skyplot_equatorial_weight1',
# #                plot_path = agn_cores_output_dir('catalogues'))











# plot_catalogue(src_ra, src_dec, src_weight, radians=False,
#                filename = '2rxs_100brightest_skyplot_equatorial',plot_path = agn_cores_output_dir('catalogues'))




# new_cat = np.empty(len(raw_cat), dtype=cat_dtype)
# new_cat["ra"] = np.deg2rad(raw_cat["RA"])  # NVSS RA in radians
# new_cat["dec"] = np.deg2rad(raw_cat["DEC"]) # NVSS DEC in radians
# new_cat["Distance (Mpc)"] = np.ones(len(raw_cat))
# new_cat["Ref Time (MJD)"] = np.ones(len(raw_cat))
#
# src_name = []
# for vv10, rxs in zip(raw_cat['NAME_vv10'], raw_cat['2RXS_ID']):
#     if (vv10!='N/A'):
#         src_name.append(vv10)
#     else:
#         src_name.append(rxs)
#
# new_cat["Name"] = src_name
# new_cat["Relative Injection Weight"] = raw_cat["2RXS_SRC_FLUX"]*1e13
#
# save_path = agn_catalogue_name("radioloud", "2rxs_100brightest_srcs")
#
# print "Saving to", save_path
#
# np.save(save_path, new_cat)
#
#
# # Plot the catalogue
# import astropy.coordinates as coord
# import astropy.units as u
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
# from matplotlib.colors import LogNorm
#
# ra = coord.Angle(np.rad2deg(new_cat['ra'])*u.degree)
# ra = ra.wrap_at(180*u.degree)
# dec = coord.Angle(np.rad2deg(new_cat['dec'])*u.degree)
# weight = new_cat['Relative Injection Weight']
#
# # plot map
# fig = plt.figure(figsize=(12.5, 7.5))
# ax = fig.add_subplot(111, projection="mollweide")
#
# sc = ax.scatter(ra.radian, dec.radian,
#                 c=weight, cmap ='magma_r',marker='.',
#                 norm=LogNorm(vmin=weight.min(), vmax=weight.max()),
#                 alpha=1, lw=0)
# clbar = plt.colorbar(sc,orientation='vertical', fraction=0.025)
# clbar.set_label(r'X-Ray flux [erg cm$^{-1}$ s$^{-1}$]', size = 20)
#
# ax.grid(1)
# ax.axes.set_axisbelow(False)
# ax.text(0.99, 0.01, 'Equatorial',
#         verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes,
#         color='black', fontsize=20)
#
# plot_path = agn_cores_output_dir('catalogues')
# filename = '2rxs_100brightest_skyplot_equatorial'
# fig.savefig(plot_path +filename + '.png', format='png',  bbox_inches="tight")
# print "Saving to", plot_path
# plt.close(fig)

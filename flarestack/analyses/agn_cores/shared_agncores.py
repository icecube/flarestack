from __future__ import print_function
from builtins import zip
import os
import numpy as np
from flarestack.shared import plot_output_dir, catalogue_dir
from flarestack.utils.catalogue_loader import load_catalogue

agncores_dir = os.path.abspath(os.path.dirname(__file__))
agncores_cat_dir = agncores_dir + "/catalogues/"
raw_cat_dir = agncores_cat_dir + "raw/"
subset_agn_dir = catalogue_dir + "agn_cores/"

try:
    os.makedirs(subset_agn_dir)
except OSError:
    pass

agn_cats = ["radioloud", "colorselected", "llang"]


def agn_cores_output_dir(name='foldername'):
    return plot_output_dir('analyses/agn_cores/'+name+'/')


def agn_catalogue_name(agn_type, xraycat='2rxs', base_dir=agncores_cat_dir):
    agn_name = agn_type + "_" + xraycat + ".npy"

    return base_dir + agn_name


def agn_subset_catalogue_name(agn_type, xray_cat, n_sources):
    return agn_catalogue_name(
        agn_type, xray_cat + "_" + str(n_sources) + "brightest_srcs",
        base_dir=subset_agn_dir
    )


def agn_subset_catalogue(agn_type, xray_cat, n_sources):
    subset_path = agn_subset_catalogue_name(agn_type, xray_cat, n_sources)
    if not os.path.isfile(subset_path):
        parent_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))
        parent_cat = np.sort(parent_cat, order="base_weight")[::-1]
        new_cat = parent_cat[:n_sources]
        print("Catalogue not found. Creating one at:", subset_path)
        np.save(subset_path, new_cat)
    return subset_path

def agn_subset_catalogue_name_north(agn_type, xray_cat, n_sources):
    return agn_catalogue_name(
        agn_type, xray_cat + "_" + str(n_sources) + "brightest_srcs_north",
        base_dir=subset_agn_dir
    )

def agn_subset_catalogue_north(agn_type, xray_cat, n_sources):
    subset_path = agn_subset_catalogue_name_north(agn_type, xray_cat, n_sources)
    if not os.path.isfile(subset_path):
        parent_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))
        parent_cat = parent_cat[parent_cat["dec_rad"]>np.deg2rad(-5)]
        parent_cat = np.sort(parent_cat, order="base_weight")[::-1]
        new_cat = parent_cat[:n_sources]
        print("Catalogue not found. Creating one at:", subset_path)
        print (new_cat)
        np.save(subset_path, new_cat)
        print ("Catalogue length is: ", len(subset_path))
    return subset_path


def agn_subset_catalogue_name_north_no_pole(agn_type, xray_cat, n_sources):
    return agn_catalogue_name(
        agn_type, xray_cat + "_" + str(n_sources) + "brightest_srcs_north_no_pole",
        base_dir=subset_agn_dir
    )

def agn_subset_catalogue_north_no_pole(agn_type, xray_cat, n_sources):
    subset_path = agn_subset_catalogue_name_north_no_pole(agn_type, xray_cat, n_sources)
    if not os.path.isfile(subset_path):
        parent_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))
        print ("Original catalogue (before north + pole + nrsrc selection) is: ", len(parent_cat))
        parent_cat = parent_cat[parent_cat["dec_rad"]>np.deg2rad(-5)]
        print("Original catalogue (after north selection) is: ", len(parent_cat))
        parent_cat = parent_cat[parent_cat["dec_rad"] < np.deg2rad(80)]
        print("Original catalogue (after north + pole selection) is: ", len(parent_cat))

        parent_cat = np.sort(parent_cat, order="base_weight")[::-1]
        new_cat = parent_cat[:n_sources]
        print("Catalogue not found. Creating one at:", subset_path)
        np.save(subset_path, new_cat)
    return subset_path


def agn_subset_catalogue_name_no_pole(agn_type, xray_cat, n_sources):
    return agn_catalogue_name(
        agn_type, xray_cat + "_" + str(n_sources) + "brightest_srcs_no_pole",
        base_dir=subset_agn_dir
    )

def agn_subset_catalogue_no_pole(agn_type, xray_cat, n_sources):
    subset_path = agn_subset_catalogue_name_no_pole(agn_type, xray_cat, n_sources)
    if not os.path.isfile(subset_path):
        parent_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))
        print ("Original catalogue (before north + pole + nrsrc selection) is: ", len(parent_cat))
        # parent_cat = parent_cat[parent_cat["dec_rad"]>np.deg2rad(-5)]
        # print("Original catalogue (after north selection) is: ", len(parent_cat))
        parent_cat = parent_cat[parent_cat["dec_rad"] < np.deg2rad(80)]
        print("Original catalogue (after pole selection) is: ", len(parent_cat))

        parent_cat = np.sort(parent_cat, order="base_weight")[::-1]
        new_cat = parent_cat[:n_sources]
        print("Catalogue not found. Creating one at:", subset_path)
        np.save(subset_path, new_cat)
    return subset_path


def agn_subset_catalogue_name_sindec_cut(agn_type, xray_cat, n_sources, sindec_cut):
    print ("In agn_subset_catalogue_sindec_cut")
    return agn_catalogue_name(
        agn_type, xray_cat + "_" + str(n_sources) + "brightest_srcs_below   _sicdec_" + str(sindec_cut),
        base_dir=subset_agn_dir
    )

def agn_subset_catalogue_sindec_cut(agn_type, xray_cat, n_sources, sindec_cut):
    subset_path = agn_subset_catalogue_name_sindec_cut(agn_type, xray_cat, n_sources, sindec_cut)
    if not os.path.isfile(subset_path):
        parent_cat = load_catalogue(agn_catalogue_name(agn_type, xray_cat))
        print ("Original catalogue (before north + pole + nrsrc selection) is: ", len(parent_cat))
        print(parent_cat["dec_rad"][:10])
        parent_cat = parent_cat[np.sin(parent_cat["dec_rad"])<sindec_cut]
        print("Original catalogue after sindec cut is: ", len(parent_cat))
        parent_cat = np.sort(parent_cat, order="base_weight")[::-1]
        new_cat = parent_cat[:n_sources]
        print("Catalogue not found. Creating one at:", subset_path)
        np.save(subset_path, new_cat)
    return subset_path


complete_cats = [
    ("radioloud", "radioselected"),
    ("radioloud", "irselected"),
    ("lowluminosity", "irselected")
]

complete_cats_north = [
    ("radioloud", "radioselected_north"),
    ("radioloud", "irselected_north"),
    ("lowluminosity", "irselected_north")
]

def create_random_src(min_distance=10, nr_sources=100):
    """Create nr_sources random sources in RA and DEC (in degree)
    :param min_distance : create sources with distance > than min_distance
    among each other (in degree)
    :param nr_sources : number of random sources to create
    """

    import astropy.coordinates as coord
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import numpy as np

    min_distance = min_distance * u.deg
    print('Minimum distance between sources is: ', min_distance,
          '\nNumber of random sources is: ', nr_sources)
    sources_ra = []
    sources_dec = []
    ra_first = np.random.uniform(-180, 180, 1)
    # dec_first = np.random.uniform(-90, 90, 1)
    dec_first = np.random.uniform(0, 90, 1)

    # print(ra_first, dec_first)
    sources_ra.append(ra_first)
    sources_dec.append(dec_first)

    while len(sources_dec) < nr_sources:
        ra = np.random.uniform(-180, 180, 1)
        dec = np.random.uniform(0, 90, 1)
        c1 = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
        c2 = SkyCoord(ra=sources_ra, dec=sources_dec, frame='icrs', unit='deg')
        sep_vect = c1.separation(c2)
        #     print('sep_vect: ', sep_vect.deg)
        greater_ = [True for i in sep_vect if i > min_distance]
        if (len(greater_) < len(sources_ra)):
            #         print('source < 10')
            #         print (sep_vect.deg)
            continue
        else:
            sources_ra.append(ra)
            sources_dec.append(dec)
    final_ra = []
    final_dec = []
    for RA, DEC in zip(sources_ra, sources_dec):
        final_ra.append(RA[0])
        final_dec.append(DEC[0])
    return final_ra, final_dec




def plot_catalogue(src_ra, src_dec, src_weight, radians=False,
                   filename='2rxs_100brightest_skyplot_equatorial',
                   plot_path=agn_cores_output_dir('catalogues')):
    '''Plot the catalogue in equatorial mollewide projection
    :param src_ra: in degree
    :param src_dec: in degree
    :param src_weight: variable for the colorbar axis. If no weight is given, all sources will be weighted equally
    :param radians: if ra and dec are given in radians, set it to True.
    :param filename:
    :param plot_path:
    :return: mollewied plot in equatorial coordinates
    '''

    import numpy as np
    import astropy.coordinates as coord
    import astropy.units as u
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from matplotlib.colors import LogNorm

    if(radians):
        src_ra  = np.rad2deg(src_ra)
        src_dec = np.rad2deg(src_dec)

    ra = coord.Angle(src_ra*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(src_dec*u.degree)
    weight = np.asarray(src_weight)

    # plot map
    fig = plt.figure(figsize=(12.5, 7.5))
    ax = fig.add_subplot(111, projection="mollweide")

    sc = ax.scatter(ra.radian, dec.radian, marker='.',
                    c=weight, cmap ='magma_r',
                    norm=LogNorm(vmin=weight.min(), vmax=weight.max()),
                    alpha=1, lw=0)
    clbar = plt.colorbar(sc, orientation='vertical', fraction=0.025)
    clbar.set_label(r'X-Ray flux [erg cm$^{-1}$ s$^{-1}$]', size = 20)

    ax.grid(1)
    ax.axes.set_axisbelow(False)
    ax.text(0.99, 0.01, 'Equatorial',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=20)

    fig.savefig(plot_path + filename + '.png', format='png',
                bbox_inches="tight")
    print("Saving to", plot_path)
    plt.close(fig)


# def agncores_limits(agn_type):

#     base = "analyses/agncores/calculate_sensitivity/"
#     path = base + agn_type + "/real_unblind/"

#     savepath = limit_output_path(path)

#     print "Loading limits from", savepath
#     with open(savepath, "r") as f:
#         res_dict = Pickle.load(f)
#     return res_dict

def scale_factor_correction(nr_srcs, dataset='ps_10_year'):
    if(dataset=="diffuse_8_year"):
        a = 0.2714841300703504
        b = -8.433793994990848
    else:
        a = 0.4275890891434705
        b = -8.532466540532349
    scale_factor = a*np.log10(nr_srcs)+b
    scale_factor_final = np.power(10, scale_factor) / 1e-9
    # if (nr_srcs>90):
    #     scale_factor_final =  np.power(10, scale_factor) / 1e-9 / 10  # in the same units as sensitivity [GeV*cm-2*s-1] at 1GeV
    # else:
    #     scale_factor_final = np.power(10,scale_factor)/1e-9 # in the same units as sensitivity [GeV*cm-2*s-1] at 1GeV
    return scale_factor_final
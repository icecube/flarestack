import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
import os
import logging
from flarestack.shared import catalogue_dir
from flarestack.utils.prepare_catalogue import cat_dtype
from flarestack.cosmo.neutrino_cosmology import define_cosmology_functions, \
    integrate_over_z, cumulative_z
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

def simulate_transient_catalogue(mh_dict, rate, resimulate=False,
                                 cat_name="random", n_entries=30,
                                 local_z=0.1, seed=None):

    tpdfs = [season.get_time_pdf() for season in mh_dict["dataset"].values()]

    data_start = min([time_pdf.sig_t0() for time_pdf in tpdfs])
    data_end = max([time_pdf.sig_t1() for time_pdf in tpdfs])

    try:
        injection_gamma = mh_dict["inj_dict"]["injection_energy_pdf"]["gamma"]
    except KeyError:
        raise Exception("No spectral index defined")

    rate_per_z, nu_flux_per_z, nu_flux_per_source, cumulative_nu_flux = \
        define_cosmology_functions(
            rate, 1 * u.erg, injection_gamma, nu_bright_fraction=1.0
    )

    n_tot = integrate_over_z(rate_per_z, zmin=0.0, zmax=8.0)
    logger.info(
        "We can integrate the rate up to z=8.0. This gives {:.3E}".format(n_tot)
    )

    n_local = integrate_over_z(rate_per_z, zmin=0.0, zmax=local_z)
    logger.info("We will only simulate up to z={0}. In this volume, there are {1:.3E}".format(local_z, n_local))

    sim_length = (data_end - data_start) * u.day

    logger.info("We simulate for {0}".format(sim_length))

    n_local = int(n_local * sim_length)

    logger.debug("Entries in catalogue {0}".format(n_local))

    logger.debug("We expect this region to contribute {:.3g} of all the flux from this source class".format(
        cumulative_nu_flux(local_z)[-1] / cumulative_nu_flux(8.0)[-1]))

    n_catalogue = sorted(list(set(
        [int(x) for x in np.logspace(-4, 0, n_entries) * n_local
         if int(x) > 0]
    )))

    cat_names_north = [catalogue_dir + cat_name + "/" + str(n) +
                       "_cat_northern.npy" for n in n_catalogue]
    cat_names_south = [catalogue_dir + cat_name + "/" + str(n) +
                       "_cat_southern.npy" for n in n_catalogue]
    cat_names = [catalogue_dir + cat_name + "/" + str(n) +
                 "_cat_full.npy" for n in n_catalogue]

    all_cat_names = {
        "Northern": cat_names_north,
        "Southern": cat_names_south,
        "Full": cat_names
    }

    if seed is not None:
        np.random.seed(seed)

    if not np.logical_and(
            np.sum([os.path.isfile(x) for x in cat_names]) == len(cat_names),
            not resimulate
    ):
        catalogue = np.empty(n_local, dtype=cat_dtype)

        catalogue["source_name"] = ["src" + str(i) for i in range(n_local)]
        catalogue["ra_rad"] = np.random.uniform(0., 2 * np.pi, n_local)
        catalogue["dec_rad"] = np.arcsin(np.random.uniform(-1., 1., n_local))
        catalogue['injection_weight_modifier'] = np.ones(n_local)
        catalogue['base_weight'] = np.ones(n_local)
        catalogue["ref_time_mjd"] = np.random.uniform(
            data_start, data_end, n_local
        )
        catalogue["start_time_mjd"] = 0.0
        catalogue["end_time_mjd"] = 0.0
        # Define conversion fraction to sample redshift distribution

        zrange = np.linspace(0, local_z, int(1e3))

        count_ints = [(x * sim_length).value
                      for x in cumulative_z(rate_per_z, zrange)]
        count_ints = np.array([0] + count_ints) / max(count_ints)

        rand_to_z = interp1d(count_ints, zrange[:-1])

        z_vals = sorted(rand_to_z(np.random.uniform(0., 1.0, n_local)))

        mpc_vals = [Distance(z=z).to("Mpc").value for z in z_vals]

        catalogue["distance_mpc"] = np.array(mpc_vals)

        dec_ranges = [
            ("Northern", 0., 1.),
            ("Southern", -1, 0.),
            ("Full", -1., 1.)
        ]

        for i, n in enumerate(n_catalogue):

            for (key, dec_min, dec_max) in dec_ranges:

                index = int(n)

                cat = catalogue[:index]

                cat_path = all_cat_names[key][i]

                mask = np.logical_and(
                    np.sin(cat["dec_rad"]) > dec_min,
                    np.sin(cat["dec_rad"]) < dec_max
                )

                try:
                    os.makedirs(os.path.dirname(cat_path))
                except OSError:
                    pass

                np.save(cat_path, cat[mask])

                logger.info("Saved to {0}".format(cat_path))

    return all_cat_names


def simulate_transients(sim_length_year, rate, injection_gamma=2.0,
                        local_z=0.1):

    rate_per_z, nu_flux_per_z, cumulative_nu_flux = define_cosmology_functions(
        rate, 1 * u.erg, injection_gamma, nu_bright_fraction=1.0
    )

    print("We can integrate the rate up to z=8.0. This gives")
    n_tot = integrate_over_z(rate_per_z, zmin=0.0, zmax=8.0)
    print("{:.3E}".format(n_tot))

    print("We will only simulate up to z=" + str(local_z) + ".")
    n_local = integrate_over_z(rate_per_z, zmin=0.0, zmax=local_z)
    print("In this volume, there are", "{:.3E}".format(n_local))

    sim_length = (sim_length_year * u.year).to("day")

    print("We simulate for", sim_length)

    n_local = int(n_local * sim_length)

    print("Entries in catalogue", n_local)

    print("We expect this region to contribute")
    print("{:.3g}".format(
        cumulative_nu_flux(local_z)[-1] / cumulative_nu_flux(8.0)[-1]))
    print("of all the flux from this source class")

    # Define conversion fraction to sample redshift distribution

    zrange = np.linspace(0, local_z, 1e3)

    count_ints = [(x * sim_length).value
                  for x in cumulative_z(rate_per_z, zrange)]
    count_ints = np.array([0] + count_ints) / max(count_ints)

    rand_to_z = interp1d(count_ints, zrange[:-1])

    z_vals = sorted(rand_to_z(np.random.uniform(0., 1.0, n_local)))

    return z_vals

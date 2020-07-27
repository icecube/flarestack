import logging
from flarestack.cosmo.rates.sfr_rates import get_sfr_rate
from flarestack.cosmo.rates.ccsn_rates import get_ccsn_rate
from flarestack.cosmo.rates.tde_rates import get_tde_rate

source_maps = {
    "tde": ["TDE", "tidal_disruption_event"],
    "sfr": ["SFR", "star_formation_rate"],
    "ccsn": ["CCSN", "sn", "supernova", "core_collapse_supernova"]
}

sources = {
    "tde": get_tde_rate,
    "sfr": get_sfr_rate,
    "ccsn": get_ccsn_rate
}

def get_rate(source_name="tde", evolution_name=None, rate_name=None, **kwargs):
    """Get rate of astrophysical object, as a function of redshift

    :param source_name: Name of source class to use
    :param evolution_name: Name of source evolution for that class to be used
    :param rate_name: Name of local rate for that class to be used
    :return:
    """

    # Check aliases

    if source_name not in sources.keys():
        new = None
        for key, maps in source_maps.items():
            if source_name in maps:
                new = key

        if new is not None:
            source_name = new
        else:
            raise Exception(f"Source class '{source_name}' not recognised. "
                            f"The following source evolutions are available: {source_name.keys()}")

    logging.info(f"Loading source class '{source_name}'")

    return sources[source_name](evolution_name, rate_name, **kwargs)
import logging
import astropy
from astropy import units as u
import numpy as np
import math
from astropy.coordinates import Distance
from flarestack.core.energy_pdf import EnergyPDF

logger = logging.getLogger(__name__)


def calculate_astronomy(flux, e_pdf_dict):
    flux /= u.GeV * u.cm**2 * u.s

    energy_PDF = EnergyPDF.create(e_pdf_dict)

    astro_res = dict()

    # phi_integral = energy_PDF.flux_integral() * u.GeV # unused

    e_integral = energy_PDF.fluence_integral() * u.GeV**2

    # Calculate fluence

    tot_fluence = flux * e_integral

    astro_res["Energy Flux (GeV cm^{-2} s^{-1})"] = tot_fluence.value

    logger.debug("Energy Flux:{0}".format(tot_fluence))

    logger.debug("Total flux: {0}".format(flux))

    logger.debug(
        "The energy range was assumed to be between {0} and {1}".format(
            energy_PDF.integral_e_min, energy_PDF.integral_e_max
        )
    )

    return astro_res

"""Results from the CCSN analysis, performed by Alexander Stasik.
Methods and results are detailed in https://edoc.hu-berlin.de/handle/18452/19442
Two methods were used, both a traditional fixed-weight analysis and a
fixed-weight analysis.
"""
from astropy import units as u
from flarestack.utils.neutrino_astronomy import calculate_astronomy
from flarestack.core.energy_PDFs import EnergyPDF
import numpy as np
# Results

limit_units = 1./(u.cm ** 2 * u.GeV)

# Limits for 1b/c (Choked jet), given for a time window of -20 to 0 days

limits = {
    "SN1bc": {
        "Fixed": 108.169653982,
        "Fit": 112.567370941
    },
    "SNIIn": {
        "Fixed": 903.46752,
        "Fit": 903.46752
    },
    "SNIIP": {
        "Fixed": 389.757926076,
        "Fit": 436.534453846
    }
}

# Limits are quoted assuming a reference distance of 10 Mpc for a source

ref_dist = 10 * u.Mpc

ref_catalogue = np.array([ref_dist.value], dtype=[("Distance (Mpc)", np.float)])

# Limits are calculated assuming the following Energy PDF

e_pdf_dict = {
    "Name": "Power Law",
    "Gamma": 2.5
}

energy_PDF = EnergyPDF.create(e_pdf_dict)
e_integral = energy_PDF.fluence_integral() * u.GeV**2

for (sn, sn_dict) in limits.iteritems():
    for llh_type in ["Fixed", "Fit"]:
        area = (ref_dist.to("cm"))**2
        energy = (sn_dict[llh_type] * limit_units * area * e_integral).to(
            "erg")
        sn_dict[llh_type + " Energy (erg)"] = energy

        print sn_dict[llh_type] * limit_units * e_integral, e_integral

print limits

if __name__ == "__main__":
    print 3.18575405845223e-09 * limit_units



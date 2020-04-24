"""Results from the CCSN analysis, performed by Alexander Stasik.
Methods and results are detailed in https://edoc.hu-berlin.de/handle/18452/19442
Two methods were used, both a traditional fixed-weight analysis and a
fixed-weight analysis.
"""
from astropy import units as u
from astropy.table import Table
from flarestack.utils.neutrino_astronomy import calculate_astronomy
from flarestack.core.energy_pdf import EnergyPDF
import numpy as np
# Results

limit_units = 1./(u.cm ** 2 * u.GeV * u.sr)

# Limits for 1b/c (Choked jet), given for a time window of -20 to 0 days

limits = {
    "Ibc": {
        "Fixed": 108.169653982,
        "Fit": 112.567370941
    },
    "IIn": {
        "Fixed": 903.46752,
        "Fit": 903.46752
    },
    "IIP": {
        "Fixed": 389.757926076,
        "Fit": 436.534453846
    }
}

# Limits are quoted assuming a reference distance of 1 Mpc for a source

ref_dist = 1 * u.Mpc

# Limits are calculated assuming the following Energy PDF

e_pdf_dict = {
    "energy_pdf_name": "power_law",
    "gamma": 2.5
}

energy_PDF = EnergyPDF.create(e_pdf_dict)
e_integral = energy_PDF.fluence_integral() * u.GeV**2

for (sn, sn_dict) in limits.items():
    for llh_type in ["Fixed", "Fit"]:
        area = (ref_dist.to("cm"))**2
        energy = (sn_dict[llh_type] * 4 * np.pi * u.sr * limit_units *
                  area * e_integral).to("erg")
        sn_dict[llh_type + " Energy (erg)"] = energy

dt = {'names': ['t', 'E'], 'formats': ['<f8', '<f8']}
dt_pvals = {'names': ['t', 'pval'], 'formats': ['<f8', '<f8']}


p_vals = {
    'box': {
        'Ibc': np.array([
            (20, 0.5), (100, 0.5), (300, 0.5), (1000, 0.348)
        ], dtype=dt_pvals),
        "IIp": np.array([
            (100, 0.317), (300, 0.5), (1000, 0.5)
        ], dtype=dt_pvals),
        "IIn": np.array([
            (100, 0.064), (300, 0.473), (1000, 0.5)
        ], dtype=dt_pvals)
    },

    'decay': {
        'IIn': np.array([
            (0.02, 0.016), (0.2, 0.426), (2., 0.5)
        ], dtype=dt_pvals),
        'IIp': np.array([
            (0.02, 0.5), (0.2, 0.5), (2., 0.5)
        ], dtype=dt_pvals)
    }
}

limits_figure_paper = {

    'box':{
        "Ibc": np.array([
            (20, 1.5e48), (100, 1.8e48), (300, 2.6e48), (1000, 4.1e48)
        ], dtype=dt),
        "IIp": np.array([
            (100, 0.363e49), (300, 0.284e49), (1000, 0.401e49)
        ], dtype=dt),
        "IIn": np.array([
            (100, 1.55e49), (300, 1.6e49), (1000, 1.53e49)
        ], dtype=dt)
    },

    'decay': {
        'IIn': np.array([
            (0.02, 2.45e49), (0.2, 1.96e49), (2., 3.1e49)
        ], dtype=dt),
        'IIp': np.array([
            (0.02, 0.355e49), (0.2, 0.263e49), (2., 1.38e49)
        ], dtype=dt)
    }
}


limits_figure_sens = {

    'box':{
        'Ibc': np.array([
            (20, 8e47), (100, 1e48), (300, 1.4e48), (1000, 1.7e48)
        ], dtype=dt),
        'IIp': np.array([
            (100, 2e48), (300, 1.8e48), (1000, 3e48)
        ], dtype=dt),
        'IIn': np.array([
            (100,6e48), (300, 7e48), (1000, 8e48)
        ], dtype=dt)
    },

    'decay': {
        'IIn': np.array([
            (0.02, 1.1e49), (0.2, 1e49), (2., 9e48)
        ], dtype=dt),
        'IIp': np.array([
            (0.02, 2.2e48), (0.2, 3e48), (2., 3.2e48)
        ], dtype=dt)
    }

}


def get_figure_limits(sn_type, pdf_type, sens=False):
    if not sens:
        dictionary = limits_figure_paper
    else:
        dictionary = limits_figure_sens

    if sn_type == 'IIP':
        return dictionary[pdf_type]['IIp']
    else:
        return dictionary[pdf_type][sn_type]


if __name__ == "__main__":
    print("Limits", limits)
    print("limits from figure", limits_figure_paper)



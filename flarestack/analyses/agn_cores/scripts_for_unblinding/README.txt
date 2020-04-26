Name: AGN Cores Analysis
Analyser: Federica Bradascio (federica.bradascio@desy.de)
Unblinding Date:
Flarestack Release: v1.1.0 (Titan)
Wiki Page: https://wiki.icecube.wisc.edu/index.php/TDE
Datasets: diffuse_8_year
Results: https://wiki.icecube.wisc.edu/index.php/Neutrinos_from_AGN_cores

This analysis uses three AGN samples: Radio-selected AGN, IR-selected AGN
and LLAGN. It performs a stacking analysis using the "fitting weights"
method and the X-ray flux as source weight.

REPRODUCIBILITY GUIDE

To create the agn catalogues, create_*agn_north_catalogue.py must be run,
one for each AGN sample.

To reproduce the 3 stacking analyses, the three scripts *agn_analysis.py
must be run. In all cases, the fit results will be obtained.

To reproduce the sensitivity curves as a function of the minimum energy
of the injected neutrinos, calculate_energy_range.py must be run.
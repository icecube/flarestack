# Search for neutrinos from AGN cores

**Name:** AGN Cores Stacking Analysis

**Analyser:** Federica Bradascio (federica.bradascio@desy.de)

**Unblinding Date:** May, 2020

**Flarestack Release:** v2.2.0 (Titan)

**Datasets:** diffuse_8_year

**Wiki Page:** https://wiki.icecube.wisc.edu/index.php/Neutrinos_from_AGN_cores

**Results:** https://wiki.icecube.wisc.edu/index.php?title=Radio_Galaxies_Stacking_Analysis/Results&action=edit&redlink=1

This analysis uses three AGN samples: Radio-selected AGN, IR-selected AGN
and LLAGN. It performs a stacking analysis using the "fitting weights"
method and the X-ray flux as source weight.

## STEPS TO REPRODUCE AGN CORES ANALYSIS

1. Create the 3 AGN samples:
    - The `scripts_for_unblinding/create_agn_samples/create_*agn_north_catalogue.py` must be run, one for each AGN sample

2. Reproduce sensitivity/dp study for each sub-sample of N brightest sources:
    - The three `scripts scripts_for_unblinding/*agn_analysis.py` must be run.
    In all cases, the fit results will be obtained.
    - To reproduce sensitivity and DP plots, the `scripts_for_unblinding/make_plots_all_samples.ipynb`
    notebook must be used.

3. (OPTIONAL) Energy range study:
    - To reproduce the sensitivity curves as a function of the minimum energy of the injected neutrinos,
    `scripts_for_unblinding/calculate_energy_range_lower.py` (lower energy bound) and
    `scripts_for_unblinding/calculate_energy_range_upper.py` (upper energy bound) must be run.
    - To reproduce energy range plots, the `scripts_for_unblinding/make_energy_range_plots.ipynb`
    notebook must be used.

4. Unbind results in terms of integral significance/upper limits:
    - Run the `scripts_for_unblinding/unblind_integral_*agn.py` for each catalogue.

5. Differential sensitivity + differential significance/upper limits (unblinding):
    - Run `scripts_for_unblinding/differential_sensitivity/diff_sens_*agn_analysis.py`
    - For UNBLINDING, run `scripts_for_unblinding/differential_sensitivity/unblind_differential_*agn.py`

All scripts are run on the DESY (Zeuthen) clusters.
All plots can be found in: `/path/to/scratch/flarestack_data/output/plots/analyses/agn_cores/`


## DESCRITPION OF SCRIPTS RELEVANT FOR UNBLINDING

- `scripts_for_unblinding/create_agn_samples/create_radio_selected_agn_north_catalogue.py`,
`scripts_for_unblinding/create_agn_samples/create_ir_selected_agn_north_catalogue.py`,
`scripts_for_unblinding/create_agn_samples/create_llagn_north_catalogue.py`:
    takes the original catalogue FITS files in /catalogues and produces the the .npy catalogue files.

- `scripts_for_unblinding/radio_selected_agn_analysis.py`, `scripts_for_unblinding/ir_selected_agn_analysis.py`,
`scripts_for_unblinding/llagn_analysis.py`:
    calculate the sensitivity/dp for 3 AGN samples, and for 10 sub-selections of X-ray brightest sources
    of each AGN sample.

- `scripts_for_unblinding/unblind_integral_radio_selected_agn.py`,
`scripts_for_unblinding/unblind_integral_ir_selected_agn.py`,
`scripts_for_unblinding/unblind_integral_llagn.py`:
    scripts to unblind the 3 AGN catalogues as integral significance/upper limits.

- `scripts_for_unblinding/make_plots_all_samples.ipynb`: notebook to create sensitivity/dp plots.

- `scripts_for_unblinding/differential_sensitivity/diff_sens_radio_selected_agn_analysis.py`,
`scripts_for_unblinding/differential_sensitivity/diff_sens_ir_selected_agn_analysis.py`,
`scripts_for_unblinding/differential_sensitivity/diff_sens_llagn_analysis.py`:
    calculate the sensitivity/dp for the 3 AGN samples (total number of sources)
    in energy decades between 100 GeV and 10 PeV.

- `scripts_for_unblinding/differential_sensitivity/unblind_differential_radio_selected_agn.py`,
`scripts_for_unblinding/differential_sensitivity/unblind_differential_ir_selected_agn.py`,
`scripts_for_unblinding/differential_sensitivity/unblind_differential_llagn.py`:
    scripts to unblind the 3 AGN catalogues as differential significance/upper limits.

- `scripts_for_unblinding/energy_range/calculate_energy_range_lower.py`,
`scripts_for_unblinding/energy_range/calculate_energy_range_upper.py`:
    calculate energy range relevant for this analysis, using 100 radio-selected AGN sources.

- `scripts_for_unblinding/make_energy_range_plots.ipynb`:
    notebook to create sensitivity/dp plots as a function of min/max energy bound.

**All other scripts are either not relevant for the unblinding or deprecated.**

**Name**: Core Collapse Supernova Analysis

**Analyser**: Jannis Necker (updated) Alex Stasik (original)

**Wiki Page**: https://wiki.icecube.wisc.edu/index.php/Point_Source_Supernova_Stacking_Analysis/Update

**Datasets**: ps_tracks (v002_p01)

**Flaresatck Version** v2.1.0 (Titan)


This analysis was originally performed by Alexander Stasik, using three CCSNe
catalogues with multiple time PDFs. Both fixed weight analyses, as
well as weight-fitting analyses, were performed. The full CCSN
catalogues were split by cumulative weight assuming an E^-2 spectrum,
with a nearby sample containing 70% of the weight being analysed
separately from the remaining 30%. Individual analyses on specific CCSNe were
also performed. 

The original supernova catalogue contained some errors. I (Jannis Necker) went through the nearby
sample and and checked each supernova and collected references. A link to the catalogue and 
references can be found on the Wiki. To also try to reproduce the sensitivities assuming the decay model, 
the corresponding PDF was implemented.


**REPRODUCICIBILITY GUIDE**

For the sensitivity calculation run `box_sensitivity.py` and `decay_sensitivity.py` respectively.

If the .npy catalogue files are not present under 
`/path/to/flaresatck/flaresatck/analyses/ccsn/necker2019/catalogues`
download the updated and flarestack readable .csv files from the link in the Wiki and store 
them under
`/path/to/flaresatck/flaresatck/analyses/ccsn/necker2019/catalogues/raw_necker/<sn_type>.csv`.
Then run `build_catalogues_from_raw.py`

The plots illustrating the difference to Alex Stasik's original results, 
the reproduced ones with _Flaresatck_ and the ones from this analysis 
are made with `make_ratio_plots.py`. Note that before that, the analysis scripts under 
`/path/to/flaresatck/flaresatck/analyses/ccsn/stasik2017/` have to be run.

All plots can be seen under 
/path/to/scratch/flarestack_data/output/plots/analyses/ccsn/stasik2107/fit_weights/


**Script Descriptions**

* `box_sensitivity.py`, `decay_sensitivity.py`: calculate the sensitivity for the box PDF and the decay PDF
* `build_catalogues_from_raw.py`: takes the original catalogue CSV files in `/catalogues` and produces the 
the `.npy` catalogue files.
* `ccsn_helpers.py`: helper functions
* `make_ratio_plots.py`: produces plots of the sensitivity ratios of the results reproduced with _Flarestack_, 
the original sensitivities and the ones produced with the corrected catalogue.
* `unblind_updated_ccsn.py`: script to unblind the analysis.
* `plot_sensitivity_fluxes.py`: loads the previously calculated sensitivities and uses `flarestack.cosmo` 
to calculate the contribution to the diffuse flux
* `compare_sensitivity_SN2009hd.ipynb`: a jupyter notebook that compares the sensitivity for the 
supernova IIP catalogue with and without SN2009hd
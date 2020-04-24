Name: Core Collapse Supernova Analysis
Analyser: Alex Stasik (original) Robert Stein (reproduced) (robert.stein@desy.de) Jannis Necker (reproduced) (jannis.necker@desy.de)
Wiki Page: https://wiki.icecube.wisc.edu/index.php/Point_Source_Supernova_Stacking_Analysis
Datasets: ps_tracks (v002_p01)

This analysis was originally performed by Alexander Stasik, using three CCSNe
catalogues with multiple time PDFs. Both fixed weight analyses, as
well as weight-fitting analyses, were performed. The full CCSN
catalogues were split by cumulative weight assuming an E^-2 spectrum,
with a nearby sample containing 70% of the weight being analysed
separately from the remaining 30%. Individual analyses on specific CCSNe were
also performed. As part of the paper drafting, I (Robert Stein) reproduced the
key results presented in that paper. The original full catalogues were provided
by Alex, and the splitting was done using convert_old_sn_catalogue.py to the new
format. Sensitivities/Discovery Potentials as a function of spectral index were
calculated using calculate_sensitivity.py, and the results were unblinded.

As in the original analysis, no significant excess was found, and resulting
upper limits were calculated accordingly.

After the results of Alex Stasik and Flarestack were found to not agree, I (Jannis Necker), did a thorough investigation.
I copied the original catalogue from the wiki page on order to be able to also reproduce
the sensitivities for the decay model. A link to the catalogue file I used can be found on the Wiki of the Update
(https://wiki.icecube.wisc.edu/index.php/Point_Source_Supernova_Stacking_Analysis/Update)
The results of the sensitivity calculation and a comparison to Alex Stasik's results can be found on the wiki.

REPRODUCICIBILITY GUIDE

If the .npy catalogue files are not present, download the original catalogues as .csv from the file referenced in the
update wiki. Copy them to /path/to/flarestack/analyses/ccsn/stasik2017/catalogues/raw and execute
build_catalogues_from_raw.py in the same directory.

For the sensitivity calculation run calculate_sensitivity_reproduce_stasik.py
(calculate_sensitivity_reproduce_stasik_decay.py respectively for the decay model sensitivities)

All plots can be found under
/path/to/scratch/flarestack_data/output/plots/analyses/ccsn/stasik2107/fit_weights/
Name: Core Collapse Supernova Analysis
Analyser: Alex Stasik (original) Robert Stein (reproduced) (robert.stein@desy.de)
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


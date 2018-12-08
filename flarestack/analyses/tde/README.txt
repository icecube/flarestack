Name: Tidal Disruption Events Analysis
Analyser: Robert Stein (robert.stein@desy.de)
Unblinding Date: 01 Nov 2018, 06 Dec 2018
Flarestack Release: v1.1.0 (Asteria), v1.2.1 (Asteria)
Wiki Page: https://wiki.icecube.wisc.edu/index.php/TDE
Datasets: ps_tracks (v002_p01) and gfu (v002_p01), gfu (v002_p04)
Results: https://drive.google.com/file/d/1nwctXeuE95lgckDjCxkrkb0ZXlFXdkch/view

This analysis used four TDE catalogues, and performed a stacking analysis used
the "fitting weights" method. It also selected four individual TDEs, and
performs a flare search analysis on each of them. The results summary can be
found in a neutrino sources call.

This analysis found no significant correlation between TDEs and neutrinos. Upper
 limits were accordingly derived, and using tde_cosmology, diffuse flux
 constraints calculated.

Shortly after unblinding, new GFU data became available. Accordingly, AT2018cow
was added as a source for the flare search, in order to provide a timely IceCube
 analysis of this source. The new data set was added to the code, and a new
 Flarestack v1.2.1 (Asteria) release was made. No significant excess was
 observed from this source.

REPRODUCIBILITY GUIDE

To reproduce the 5 flare searches, unblind_individual_tdes.py must be run. To
reproduce the 4 stacking analyses, unblind_tde_catalogues.py must be run. In
both cases, the fit results will be obtained.

To reproduce the sensitivity curves as a function of spectral index,
compare_spectral_indices.py must be run. For the comparison of time integration
and flare search methods, compare_cluster_search_to_time_integration.py must be
run. Both of these scripts should be run on the cluster to obtain sufficient
statistics.

If the unblinding scripts are run *after* the sensitivity scripts have been run,
 TS distributions will be available. Then the TS results for each unblinding
 will automatically be compared to background TS distributions, and the
 significance of the result will be quantified. Limits will also automatically
 be set. All plots can be viewed in the assigned scratch directory, under
  /path/to/scratch/flarestack_data/output/plots/analyses/tde/....

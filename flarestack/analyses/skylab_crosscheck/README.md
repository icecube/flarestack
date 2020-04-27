**Name**: Stacking behavior crosscheck with SkyLab

**Analyser**: Jannis Necker (jannis.necker@desy.de)

**Flarestack version**: v2.0.2 (Titan)

**Wiki Page**: https://wiki.icecube.wisc.edu/index.php/Flarestack/Skylab_crosscheck


This crosscheck compares the stacking behavior of Flaresatck and SkyLab. 
The same sources were injected with both codes
at different declinations and then successively stacked. 
The SkyLab results were obtained with version v2-08 
(see https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/skylab).
The sources are stored in `/data`, the SkyLab results in `/skylab_results`. 
Note that when running `make_sources.py` new sources will be created and 
therefore the SkyLab results have to be re-calculated as well!

**Reproducibility guide:** 

run `calculate_skylab_sensitivity.py`.
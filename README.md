# flarestack
Code for unbinned likelihood analysis of astroparticle physics data.

Both time-dependent and time-independent analyses can be performed, as well as a "flare-search" algorithm to find event clustering in time as well as space.

Perform single-source analyses, as well as stacking of sources according to predefined weighting. 
Also performs stacking analyses where the signal strength of each source is fit individually.

# Installation instructions

Flarestack uses python 2.7, and requires the following packages:

* numpy
* scipy
* astropy
* healpy
* matplotlib
* numexpr

All required dependencies can be found using the IceCube py2-v3 environment.

Flarestack also requires data and MC. IceCube users can find files in the shared /data/ana/ directory in .npy format.

Once git cloned, you must edit the config.py file, and specify a directory for the storage and output of Flarestack. This should ideally be scratch space. You can then run setup.py. Check that the path specified is the correct one. 

The setup script will create a directory structure and create files which can be reused many times, such as energy PDF splines and generic point source catalogues at various declinations. This removes the need to recalculate the same thing many times.

One additional file that is required is the published IceCube 7 year Point source sensitivity. In general, Flarestack calculates sensitivity/discovery potential by generating TS distributions at fixed points between 0 and a characteristic flux level, and then interpolates between these points to determine the 90% sensitivity and 50% discovery potential. To meanigfully estimate the sensitivity/discovery potential, the characteristic flux scale must be large enough to exceed the sensitivity or discovery potential, while still small enough so that the interpolation points are not entirely composed of overfluctutations. Selecting reasonable values for this flux scale is typically done by multiplying the published ps sensitivity by various multiplicative factors. 

The file can be downloaded from https://icecube.wisc.edu/~coenders/sens.npy, and should be saved to /path/to/scratch/flarestack__data/input/skylab_reference/sens.npy .

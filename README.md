# flarestack
Code for unbinned likelihood analysis of astroparticle physics data, created by [@robertdstein](https://github.com/robertdstein).

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

All required dependencies can be found using the IceCube py2-v3 environment. They can collectively be installed with _pip install -r requirements.txt_.

## How do I actually install Flarestack?

The answer to this question depends on how lazy you're feeling, and how much of the backend you want to deal with.

### I only want to do an analysis, and trust the under-the-hood code

In that case:
 * _pip install flarestack_
 
The entire package can simply be pip installed, and this will automatically install all dependencies.

 ### Actually, I want to see the backend code myself. Maybe I want to contribute to it!
 
 Now you will need one extra line of code:
 * _git clone git@github.com:robertdstein/flarestack.git_
 * _pip install -e flarestack_
 
### Right, I've now downloaded Flarestack. Can I use it right away?
 
Unfortunately, you can't do Science quite yet. There is an additional step, in which multiple recyleable values are computed once, and will be frequently reused later on. To perform this step, you need to firstly download some data+MC. That can be bypassed if you are working on the DESY /afs/ filesystem, or on the cobalt machinces at WIPAC, because Flarestack will instead find the relevant IceCube files automatically. Either way, you then need to select a space to save these precomputed/stored values. Since they can be regenerated at any time, a scratch space is ideal for this purpose. You then need to run a script such as the following:

  _from flarestack.build import run_setup set_scratch_directory_
  
  _set_scratch_directory("/path/to/my/scratch/")_
  
  If you are *not* using the DESY/WIPAC datasets, you'll need to add the following lines to your code:
  
  _from flarestack.shared import set_dataset_dir_
  
  _set_dataset_dir("/path/to/datasets")_
  
  In any case, you then need the next to lines to begin setup:

  _from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1_

  _run_setup(txs_sample_v1)_

The above 4/6 lines of code will then build all relevant files for the datasets used in the TXS analysis (7 years of Point Source data and 2.5 years of GFU data). *Be prepared that this will take some time*. Fortunately, having done this once, you will not need to repeat it unless you require new datasets or a new release of Flarestack. The scratch directory will not need to be set again, although the dataset directory will need to be newly assigned. 

You can them simply run scripts such as those under /flarestack/analyses/, and do your science!

One additional file that is required is the published IceCube 7 year Point source sensitivity. In general, Flarestack calculates sensitivity/discovery potential by generating TS distributions at fixed points between 0 and a characteristic flux level, and then interpolates between these points to determine the 90% sensitivity and 50% discovery potential. To meaningfully estimate the sensitivity/discovery potential, the characteristic flux scale must be large enough to exceed the sensitivity or discovery potential, while still small enough so that the interpolation points are not entirely composed of overfluctutations. Selecting reasonable values for this flux scale is typically done by multiplying the published ps sensitivity by various factors. 

The file can be downloaded from https://icecube.wisc.edu/~coenders/sens.npy, and should be saved to /path/to/scratch/flarestack__data/input/skylab_reference/sens.npy .

# flarestack
Code for unbinned likelihood analysis of astroparticle physics data, created by [@robertdstein](https://github.com/robertdstein).

Both time-dependent and time-independent analyses can be performed, as well as a "flare-search" algorithm to find event clustering in time as well as space.

Performs single point source analyses, as well as the stacking of sources according to predefined weighting. 
Also performs stacking analyses where the signal strength of each source is fit individually.

# Installation instructions

## How do I actually install Flarestack?

The answer to this question depends on how lazy you're feeling, and how much of the backend you want to deal with.

### OPTION A: I only want to do an analysis, and trust the under-the-hood code

In that case:
```bash
pip install flarestack
```
 
The entire package can simply be pip installed, and this will automatically install all dependencies.

 ### OPTION B: Actually, I want to see the backend code myself. Maybe I want to contribute to it!
 
 Now you will need one extra line of code:

```bash
git clone git@github.com:robertdstein/flarestack.git
pip install -e flarestack
```
 
This will give you the very latest copy of the code, update the installed version if you git pull or modify scripts yourself, and still enable you to import flarestack.

### What actually are the dependencies, by the way?

Flarestack uses python 2.7, and requires the following packages:

* numpy
* scipy
* astropy
* healpy=1.10.3
* matplotlib
* numexpr

All required dependencies can be found using the IceCube py2-v3 environment. They can collectively be installed with ```pip install -r requirements.txt```, if for some reason you don't want to install flarestack via pip..
 
### Right, anyway, I've now downloaded Flarestack. Can I use it right away?
 
Unfortunately, you can't do Science quite yet. There is an additional step, in which multiple recycleable values are computed once, and will be frequently reused later on. To perform this step, you need to firstly download some data+MC. That can be bypassed if you are working on the DESY /afs/ filesystem, or on the cobalt machinces at WIPAC, because Flarestack will instead find the relevant IceCube files automatically. Either way, you then need to select a space to save these precomputed/stored values. Since they can be regenerated at any time, a scratch space is ideal for this purpose. You then need to run a script such as the following:

```python
 from flarestack.precompute import run_precompute, set_scratch_directory
 set_scratch_directory("/path/to/my/scratch/")
```

If you are *not* using the DESY/WIPAC datasets, you'll need to add the following lines to your code:

```python
from flarestack.shared import set_dataset_dir
set_dataset_dir("/path/to/datasets/")
```

In any case, you then need the next to lines to begin setup:

 ```python
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
run_precompute(txs_sample_v1)
```

The above 4/6 lines of code will then build all relevant files for the datasets used in the TXS analysis (7 years of Point Source data and 2.5 years of GFU data). *Be prepared that this will take some time*. Fortunately, having done this once, you will not need to repeat it unless you require new datasets or a new release of Flarestack. The scratch directory will not need to be set again, although the dataset directory will need to be newly assigned at the top of your analysis scripts. 

You can them simply run scripts such as those under /flarestack/analyses/, and do your science!

### Anything else?

One additional file that is useful is the published IceCube 7 year Point source sensitivity. In general, Flarestack calculates sensitivity/discovery potential by generating TS distributions at fixed points between 0 and a characteristic flux level, and then interpolates between these points to determine the 90% sensitivity and 50% discovery potential. To meaningfully estimate the sensitivity/discovery potential, the characteristic flux scale must be large enough to exceed the sensitivity or discovery potential, while still small enough so that the interpolation points are not entirely composed of overfluctutations. Selecting reasonable values for this flux scale is typically done by multiplying the published ps sensitivity by various factors. The file can be downloaded from https://icecube.wisc.edu/~coenders/sens.npy, and should be saved to /path/to/scratch/flarestack__data/input/skylab_reference/sens.npy . As with the datasets, this step does not need to be performed for users of DESY/WIPAC because Flarestack will automatically find the relevant files. Users without access to these files will be unable to use the _reference_sensitivity_ function, but will otherwise be able to perform analyses as usual.

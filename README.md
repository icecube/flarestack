# flarestack
[![Documentation Status](https://readthedocs.org/projects/flarestack/badge/?version=latest)](https://flarestack.readthedocs.io/en/latest/?badge=latest) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IceCubeOpenSource/flarestack/master)

Code for unbinned likelihood analysis of astroparticle physics data, created by [@robertdstein](https://github.com/robertdstein).

Both time-dependent and time-independent analyses can be performed, as well as a "flare-search" algorithm to find event clustering in time as well as space.

Performs single point source analyses, as well as the stacking of sources according to predefined weighting. 
Also performs stacking analyses where the signal strength of each source is fit individually.

# Getting started

The easiest way to start using Flarestack is to play with the introductory ipython notebooks, which can be opened with the following link:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IceCubeOpenSource/flarestack/master)

The notebooks themselves are found under *examples/ipython_notebooks/*.

The "Binder" provides a pre-built Docker image containing all necessary dependencies, so you can simply click and play. It avoids the need for local installation, and should provide understanding of how the code works. 

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
 
 Now you will need a couple of extra code lines:

```bash
git clone git@github.com:IceCubeOpenSource/flarestack.git
export PYTHONPATH=/path/to/flarestack
```
 
This will give you the very latest copy of the code, update the installed version if you git pull or modify scripts yourself, and still enable you to import flarestack.

### What actually are the dependencies, by the way?

Flarestack uses python 3.7, and requires the following packages:

* numpy
* scipy
* astropy
* healpy=1.10.1
* matplotlib
* numexpr

All required dependencies can be found using the IceCube py3-v4 environment. They can collectively be installed with ```pip install -r requirements.txt```, if you don't want to install flarestack via pip.
 
### Right, anyway, I've now downloaded Flarestack. Can I use it right away?
 
Unfortunately, you can't do science quite yet. There is an additional step, in which multiple recycleable values are computed once, and will be frequently reused later on. To perform this step, you need to firstly download some data+MC. IceCube users can download these files from Madison under ```/data/ana/analyses/```, where the appropriate directory naming convention is then mirrored by the ```flarestack/data/IceCube/``` structure. That can be bypassed if you are working on the DESY /afs/ filesystem, or on the cobalt machinces at WIPAC, because Flarestack will instead find the relevant IceCube files automatically. Either way, you then need to select a space to save these precomputed/stored values. Since they can be regenerated at any time, a scratch space is ideal for this purpose. 

The most straighforward way to set up Flarestack is to run the following command from inside the flarestack directory.

```bash
python flarestack/precompute.py -sp /path/to/scratch
```

Alternatively, you can setup your own precomputation in a python console, in the following way. Firstly, you specify the scratch directory you want to use:

```python
 from flarestack.precompute import run_precompute, set_scratch_directory
 set_scratch_directory("/path/to/my/scratch/")
```

If you are *not* using the DESY/WIPAC datasets, you'll need to point to the relevant directory where raw datasets are stored. Run this command in bash:

```bash
export FLARESTACK_DATASET_DIR=/path/to/datasets/
```

Finally, you then need to select datasets and run the begin setup:

 ```python
from flarestack.data.icecube.gfu.gfu_v002_p01 import txs_sample_v1
run_precompute(txs_sample_v1)
```

The above 4/6 lines of code will then build all relevant files for the datasets used in the TXS analysis (7 years of Point Source data and 2.5 years of GFU data), in the same way as when flarestack/precompute.py is run directly.

In either case, the code will then run. *Be prepared that this will take some time*. Fortunately, having done this once, you will not need to repeat it unless you require new datasets or a new release of Flarestack. The scratch directory will not need to be set again, although the dataset directory will need to be newly assigned at the top of your analysis scripts. 

You can them simply run scripts such as those under /flarestack/analyses/, and do your science!

# Testing Flarestack

Is flarestack actually working? If you've already run the precomputation, you can check the functionality of flarestack with *unit tests*. There are a suite of unit tests to cover flarestack functionality, which can be run from the base flarestack directory with:

 ```bash
 python -m unittest discover tests/
```

If you want to contribute to flarestack, please remember to add new tests!


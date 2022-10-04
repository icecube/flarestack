# Datasets
*flarestack* is designed to work with different types of datasets.

Datasets are stored under the *flarestack* data directory (`$FLARESTACK_DATA_DIR`). Note that this is different from the `flarestack__data` directory that is automatically created under `$FLARESTACK_SCRATCH_DIR`. The former is a static repository of datasets, the latter is the actual working directory of *flarestack*. Python modules acting as interfaces to the stored datasets are included under `flarestack/data`.

## Dataset index
*flarestack* currently implements a dataset index, an auxiliary dictionary that allows to retrieve datasets by name (instead of having to look up an object in the corresponding interface module). You can access the index by importing `flarestack.data.dataset_index'. You can use it by following this example:

```python
from flarestack.data.dataset_index import dataset_index
print(dataset_index.get_dataset_list())
dataset_name = dataset_index.get_dataset_list()[0] # just get the first dataset name the list
dataset = dataset_index.get_dataset(dataset_name)
```

## Reduce a dataset to the relevant seasons
A dataset is usually composed of different seasons. When conducting time-dependent analyses, it could be more efficient to discard the season that do not overlap with the time frame of the chosen signal injection and search. The module `flarestack.utils.custom_dataset` comes to help:

```python
dataset = dataset_index.get_dataset(dataset_name)
catalogue = np.load(catalogue_path)
common_time_pdf = { "time_pdf_name": "custom_source_box" } # example time PDF

from flarestack.utils.custom_dataset import custom_dataset
reduced_dataset = custom_dataset(dataset, catalogue, common_time_pdf)
```

## Adding a new dataset
To add a new dataset to *flarestack*:
- store the corresponding files under `$FLARESTACK_DATA_DIR`. If the dataset is a new version of an existing one, follow the same directory hierarchy. Otherwise, you will likely have to create your own path specification;
- create an interface module under `flarestack/data`;
- import the corresponding dataset object in `flarestack/data/__init__.py`.

To add the dataset to the index, first import the index in the dataset interface module:
```python
from flarestack.data.dataset_index import dataset_index

sample_name = "ps_tracks_v004_p02" # give the dataset a meaningful name
ps_v004_p02 = IceCubeDataset() # instantiate the dataset
"""
[...] dataset is populated here [...]
"""
dataset_index.add_dataset("icecube." + sample_name, ps_v004_p02) # add the dataset to the index
```

**Important**: for the correct population of the index, the dataset needs to be added to `flarestack/data/__init.py__` (see above).

""" This module provides the functionality to create a dataset index by instantiating a DatasetIndex object and importing all the available datasets. Each dataset, in turns, is expect to import `dataset_index` from this module, and adding its own information.
"""

import logging
from typing import List
from flarestack.data import Dataset

logger = logging.getLogger(__name__)


class DatasetIndex:
    """Class storing an index for available datasets"""

    def __init__(self) -> None:
        self.index = dict()

    def add_dataset(self, name: str, object: Dataset):
        """adds a dataset to the index

        Args:
            name (str): assigned name of the dataset
            object (Dataset): dataset object
        """
        self.index[name] = object

    def add_alias(self, alias: str, name: str):
        """adds an alias for a dataset

        Args:
            alias (str): alias name
            name (str): dataset name
        """
        dest = self.index[name]
        if isinstance(dest, Dataset):
            self.index[alias] = name
        else:
            logger.warning("f{name} is already an alias, aliasing {dest} instead.")
            self.index[alias] = dest

    def get_dataset(self, name: str) -> Dataset:
        """retrieve a dataset by name

        Args:
            name (str): dataset name

        Returns:
            Dataset: dataset
        """
        dest = self.index[name]
        if isinstance(dest, Dataset):
            return dest
        else:
            logger.info(f"{name} is an alias for {dest}")
            return self.index[dest]

    def get_dataset_list(self) -> List[str]:
        """Get list of indexed datasets"""
        return self.index.keys()


dataset_index = DatasetIndex()

import flarestack.data.public
import flarestack.data.icecube

import unittest

from flarestack.data.dataset_index import dataset_index


class TestDatasetIndex(unittest.TestCase):
    def setUp(self):
        pass

    def test_dataset_index(self):
        for key in dataset_index.get_dataset_list():
            dataset = dataset_index.get_dataset(key)
            self.assertTrue(dataset.name in key)


if __name__ == "__main__":
    unittest.main()

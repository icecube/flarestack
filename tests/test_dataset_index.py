import unittest

from flarestack.data.dataset_index import dataset_index


class TestDatasetIndex(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_index_content(self) -> None:
        self.assertGreater(len(dataset_index.get_dataset_list()), 0)

    def test_index_consistency(self) -> None:
        for key in dataset_index.get_dataset_list():
            dataset = dataset_index.get_dataset(key)
            self.assertEqual(key, dataset.name)


if __name__ == "__main__":
    unittest.main()

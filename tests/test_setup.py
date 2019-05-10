from __future__ import print_function
import os
from flarestack.precompute import set_scratch_directory, run_precompute,\
    config_path
import unittest
import shutil


class TestSetup(unittest.TestCase):

    def setUp(self):
        with open(config_path, "r") as f:
            scratch_path = f.readline()[16:-2]
        self.addCleanup(set_scratch_directory, scratch_path)

    def test_setup(self):
        print("\n")
        print("\n")
        print("Testing setup script")
        print("\n")
        print("\n")

        temp_scratch_dir = os.path.abspath(os.path.dirname(__file__))
        self.addCleanup(self.clear_temp_dir, temp_scratch_dir)

        print("Temporary scratch directory:", temp_scratch_dir)

        set_scratch_directory(temp_scratch_dir)
        run_precompute(temp_scratch_dir, ask=False)

    @staticmethod
    def clear_temp_dir(path):
        path += "/flarestack__data"
        print("Removing", path)
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
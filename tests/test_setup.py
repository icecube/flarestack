from __future__ import print_function
import os
import unittest
import shutil
import sys


class TestSetup(unittest.TestCase):

    def setUp(self):
        pass

    def test_setup(self):
        print("\n")
        print("\n")
        print("Testing setup script")
        print("\n")
        print("\n")

        keys = list(sys.modules.keys())
        for key in keys:
            if "flarestack" in key:
                del sys.modules[key]

        from flarestack.precompute import set_scratch_directory, run_precompute, \
            config_path

        with open(config_path, "r") as f:
            scratch_path = f.readline()[16:-2]
        self.addCleanup(set_scratch_directory, scratch_path)

        temp_scratch_dir = os.path.abspath(os.path.dirname(__file__))
        self.addCleanup(self.clear_temp_dir, temp_scratch_dir)

        print("Temporary scratch directory:", temp_scratch_dir)

        set_scratch_directory(temp_scratch_dir)

        from flarestack.data.icecube.ps_tracks.ps_v002_p01 import IC40_dict
        run_precompute([IC40_dict], ask=False)

        keys = list(sys.modules.keys())
        for key in keys:
            if "flarestack" in key:
                del sys.modules[key]

    @staticmethod
    def clear_temp_dir(path):
        path += "/flarestack__data"
        print("Removing", path)
        shutil.rmtree(path)

        try:
            del sys.modules["flarestack.shared"]
        except KeyError:
            pass

        keys = list(sys.modules.keys())
        for key in keys:
            if "flarestack" in key:
                del sys.modules[key]


if __name__ == '__main__':
    unittest.main()
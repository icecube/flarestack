"""Script to convert TDE model from https://arxiv.org/abs/1904.07999,
with tabulated values kindly provided by Leonel Morejon.
These values are converted to a spline, and saved.
"""
import numpy as np
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)

path = "/Users/robertstein/Downloads/Neutrinos_from_TDE.pickle"

data = np.load(path, allow_pickle=True, fix_imports=True)

e = data["E_GeV"][:,0]
y = data["E_GeV"][:,1]

flux = y / e**2

x_vals = np.log10(e)
y_vals = np.log(flux)

f = InterpolatedUnivariateSpline(x_vals, y_vals)

# output_path =
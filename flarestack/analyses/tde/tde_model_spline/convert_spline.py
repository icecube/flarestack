"""Script to convert TDE model from https://arxiv.org/abs/1904.07999,
with tabulated values kindly provided by Leonel Morejon.
These values are converted to a spline, and saved.
"""
import numpy as np
import pickle
import os
from scipy.interpolate import InterpolatedUnivariateSpline

dir_path = os.path.dirname(os.path.realpath(__file__))

path = os.path.join(dir_path, "Neutrinos_from_TDE.pickle")

with open(path, "rb") as f:
    data = pickle.load(f, encoding='bytes')

e = data["E_GeV"][:,0]
y = data["E_GeV"][:,1]

flux = y / e**2

# x_vals = np.log10(e)
x_vals = e
y_vals = np.log(flux)

f = InterpolatedUnivariateSpline(x_vals, y_vals)

tde_spline_output_path = os.path.join(dir_path, "morejon_model_spline.pkl")

print("Saving to {0}".format(tde_spline_output_path))

with open(tde_spline_output_path, "wb") as file:
    pickle.dump(f, file)

min_nu_e_gev = min(x_vals)
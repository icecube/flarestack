"""Script to constrain TDE model from https://arxiv.org/abs/1904.07999,
with tabulated values kindly provided by Leonel Morejon.

These values are converted to a spline, and saved. This script is then run
to determine the sensitivity to these models.
"""
from scipy.interpolate import InterpolatedUnivariateSpline


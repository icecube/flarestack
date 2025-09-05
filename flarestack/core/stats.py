import numpy as np
from scipy.stats import rv_continuous


class _king_gen(rv_continuous):
    """
    King function, used to parameterize the PSF in XMM and Fermi

    See: http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html
    """

    def _argcheck(self, shape):
        return np.asarray(shape) > 1

    def _pdf(self, x, shape):
        return x * (1.0 - 1.0 / shape) * (1 + x**2 / (2.0 * shape)) ** -shape

    def _logpdf(self, x, shape):
        return (
            np.log(x) + np.log1p(-1.0 / shape) - shape * np.log1p(x**2 / (2.0 * shape))
        )

    def _cdf(self, x, shape):
        x2 = x**2
        a = 2 * shape
        return (1.0 - 1.0 / shape) / (a - 2) * (a - (a + x2) * (x2 / a + 1) ** -shape)


king = _king_gen(name="king", a=0.0)

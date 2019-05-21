import numpy as np

gfu_binning = (
    np.unique(np.concatenate([
        np.linspace(-1., -0.93, 4 + 1),
        np.linspace(-0.93, -0.3, 10 + 1),
        np.linspace(-0.3, 0.05, 9 + 1),
        np.linspace(0.05, 1., 18 + 1),
    ])),
    np.arange(1., 9.5 + 0.01, 0.125)
)
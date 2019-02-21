"""
This file contains functions to calculate the B field of 1D currents and their combinations.
"""

import numpy as np
from scipy.constants import mu_0, pi


def one_dim_current(I, phi1, phi2, R):
    B = mu_0 / (4 * pi) * I / R * (np.cos(phi1) - np.cos(phi2))
    return B

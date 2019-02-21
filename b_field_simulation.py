"""
This file contains functions to calculate the B field of 1D currents and their combinations.
"""

import numpy as np
from scipy.constants import mu_0, pi


def one_dim_current(r, x1, x2, j, d):
    """
    r: location, where the B field is probed
    x1: location of current entry
    x2: location of current exit
    j: current
    d: shortest distance of infinite current line and r
    :return: B, the magnetic field vector
    """
    phi1, phi2 = calc_angles(r, x1, x2)
    B_abs = mu_0 / (4 * pi) * j / d * (np.cos(phi1) - np.cos(phi2))
    direction = np.cross((x2 - x1), (r - x1))
    direction_norm = direction / np.linalg.norm(direction)
    return B_abs * direction_norm


def calc_angles(r, x1, x2):
    phi1 = sign(r[2]) * calc_single_angle(x2 - x1, r - x1)
    phi2 = sign(r[2]) * calc_single_angle(x2 - x1, r - x2)
    return phi1, phi2


def sign(x):
    if x >= 0:
        sig = 1
    else:
        sig = -1
    return sig


def calc_single_angle(v1, v2):
    return np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def distance_point_line(r, x1, x2):
    numerator = np.linalg.norm(np.cross(x2 - x1, x1 - r))
    denominator = np.linalg.norm(x2 - x1)
    return numerator / denominator


if __name__ == '__main__':
    r = np.array([1, 1, 0])
    x1 = np.array([0, 0, -1])
    x2 = np.array([0, 0, 1])
    print(distance_point_line(r, x1, x2))

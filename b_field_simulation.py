"""
This file contains functions to calculate the B field of 1D currents and their combinations.
"""

import numpy as np
from scipy.constants import mu_0, pi


def one_dim_current(r, x1, x2, j):
    """
    r: location, where the B field is probed
    x1: location of current entry
    x2: location of current exit
    j: current
    :return: B, the magnetic field vector
    """
    d = distance_point_line(r, x1, x2)
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

    import matplotlib.pyplot as plt

    xs = np.linspace(-10, 10, 100)
    ys = np.linspace(-10, 10, 100)

    bs_x = np.zeros((len(ys), len(xs)))
    bs_y = np.zeros((len(ys), len(xs)))
    bs_z = np.zeros((len(ys), len(xs)))
    j = 10

    x1 = np.array([0, -10, 0])
    x2 = np.array([0, 10, 0])

    for x_i, x in enumerate(xs):
        for y_i, y in enumerate(ys):
            b = one_dim_current(r=np.array([x, y, 1]), x1=x1, x2=x2, j=j)
            bs_x[y_i, x_i] = b[0]
            bs_y[y_i, x_i] = b[1]
            bs_z[y_i, x_i] = b[2]

    plt.close('all')
    plt.imshow(bs_z, aspect='auto')
    plt.colorbar()
    plt.show()

    plt.close('all')
    plt.imshow(bs_x, aspect='auto')
    plt.colorbar()
    plt.show()

    plt.close('all')
    plt.imshow(bs_y, aspect='auto')
    plt.colorbar()
    plt.show()

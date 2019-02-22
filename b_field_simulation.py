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

def quadrupole_field_2d(r, r0, axes, gradient):
    """
    r: location (3x1 array, (m))
    r0: point on axis of zero field (3x1 array, (m))
    axes: triad defining the quadrupole (m)
         (3x3 matrix with column vectors describing
          axis of 0 field, gradient 1 gradient 2 )
         axes have to be orthogonal and normalized
    gradient: gradient strength, scalar (T/m). Field grows to positive numbers 
              with this gradient along axes[:,1]
              grows to negative numbers along axes[:,2]
    """
    dr = r - r0
    dr_gradient_frame = np.dot(np.linalg.inv(axes), dr)
    B_gradient_frame = np.array((0, gradient*dr_gradient_frame[1], -gradient*dr_gradient_frame[2]))
    B_lab_frame = np.dot(axes, B_gradient_frame)
    return B_lab_frame

#%% compute field of a wire
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

    
#%% plot field of a wire 
    plt.figure()
    plt.quiver(bs_x, bs_y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
#%% compute quadrupole field
    r0 = np.array((0,0,0))
    
    axes = np.array((
            (1,0,0),
            (0,0,1),
            (0,1,0))
            )
        
    dBdr = 1
    
    for x_i, x in enumerate(xs):
        for y_i, y in enumerate(ys):
            b = quadrupole_field_2d(r=np.array([x, y, 1]), r0=r0, axes=axes, gradient = dBdr)
            bs_x[y_i, x_i] = b[0]
            bs_y[y_i, x_i] = b[1]
            bs_z[y_i, x_i] = b[2]

    
#%% plot quadrupole field
    plt.figure()
    plt.quiver(bs_x, bs_y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
#    plt.figure()
#    plt.imshow(np.sqrt(bs_x**2 + bs_y**2))
#    plt.show()
#    
#    plt.figure()
#    plt.plot(bs_x[:,50])
    
# this file contains the functions shared by numpy and scipy backend
# speically designed for ISDF
# to avoid mutual import

import numpy


def distance_translation(pa: numpy.ndarray, pb: numpy.ndarray, a: numpy.ndarray):
    """
    Calculate the distances between multiple points pa and pb, taking periodic boundary conditions into account.

    :param pa: Coordinates of points a, shape (n, 3)
    :param pb: Coordinates of points b, shape (m, 3)
    :param a: Lattice vectors (assumed to be a 3x3 array)
    :return: Minimum distances considering periodic boundary conditions, shape (n, m)
    """
    # Reshape pa and pb for broadcasting
    pa = numpy.asarray(pa)
    pb = numpy.asarray(pb)
    pa = pa[:, numpy.newaxis, :]  # Shape becomes (n, 1, 3)
    pb = pb[numpy.newaxis, :, :]  # Shape becomes (1, m, 3)

    # Calculate differences
    diff = pa - pb  # Shape becomes (n, m, 3)

    # Apply periodic boundary conditions
    for i in range(3):
        diff[:, :, i] = numpy.minimum.reduce(
            [
                numpy.abs(diff[:, :, i]),
                numpy.abs(diff[:, :, i] - a[i, i]),
                numpy.abs(diff[:, :, i] + a[i, i]),
            ]
        )

    # Calculate Euclidean distances
    distances = numpy.sqrt(numpy.sum(diff**2, axis=2))  # Shape becomes (n, m)

    return distances


def add_to_indexed_submatrix_(
    a: numpy.ndarray, idx: numpy.ndarray, idy: numpy.ndarray, b: numpy.ndarray
) -> numpy.ndarray:
    a[idx[:, numpy.newaxis], idy] += b
    return a


def copy_indexed_submatrix(
    a: numpy.ndarray, idx: numpy.ndarray, idy: numpy.ndarray, out: numpy.ndarray = None
) -> numpy.ndarray:
    if out is None:
        out = numpy.empty((len(idx), len(idy)), dtype=a.dtype)
    out[:] = a[idx[:, numpy.newaxis], idy]


def cast_to_complex(a: numpy.ndarray) -> numpy.ndarray:
    if a.dtype == numpy.complex128:
        return a
    return a.astype(numpy.complex128)

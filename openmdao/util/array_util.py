""" Some useful array utilities. """

import sys
from six.moves import range
import numpy as np
from numpy import ndarray
from itertools import product


def array_idx_iter(shape):
    """
    Return an iterator over the indices into a n-dimensional array.

    Args
    ----
    shape : tuple
        shape of the array.
    """
    for p in product(*[range(s) for s in shape]):
        yield p


def evenly_distrib_idxs(num_divisions, arr_size):
    """
    Given a number of divisions and the size of an array, chop the array up
    into pieces according to number of divisions, keeping the distribution
    of entries as even as possible.

    Args
    ----
    num_divisions : int
        Number of parts to divide the array into.

    arr_size : int
        Number of entries in the array.

    Returns
    -------
    tuple
        a tuple of (sizes, offsets), where sizes and offsets contain values for all
        divisions.
    """
    base = arr_size / num_divisions
    leftover = arr_size % num_divisions
    sizes = np.ones(num_divisions, dtype="int") * base

    # evenly distribute the remainder across size-leftover procs,
    # instead of giving the whole remainder to one proc
    sizes[:leftover] += 1

    offsets = np.zeros(num_divisions, dtype="int")
    offsets[1:] = np.cumsum(sizes)[:-1]

    return sizes, offsets

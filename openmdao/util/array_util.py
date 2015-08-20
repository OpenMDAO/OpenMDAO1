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


class SubArray(object):
    """
    A compact representation of indices into an array.  Rather than
    storing a full array index that contains all indices, this
    will contain a list of tuples indicating slices into the array
    if possible. Otherwise it will just store the index array.

    Args
    ----
    data : index array or slice

    """
    def __init__(self, data):
        self._idx = None

        if isinstance(data, ndarray):
            slc = self._idx_arr_to_slice(data)
            if slc is not None:
                self._idx = slc
        elif isinstance(data, slice):
            if data.step < 1:
                raise ValueError("invalid step value for SubArray slice (%d)" %
                                 data.step)
            self._idx = data
        else:
            raise TypeError("Can't create SubArray object using data of type %s"
                            % type(data))

    def __contains__(self, idx):
        """
        Return True if the given index is contained in the set of
        indices represented by this object.
        """
        if isinstance(self._idx, slice):
            slc = self._idx
            if idx >= slc.start and idx < slc.stop:
                # might be in this slice.
                return ((idx-slc.start) % slc.step) == 0
            return False
        else:
            return idx in self._idx

    def __iter__(self):
        if isinstance(self._idx, slice):
            for i in range(self._idx.start, self._idx.end, self._idx.step):
                yield i
        else:
            for idx in self._idx:
                yield idx

    def idx(self):
        """
        Returns
        -------
        slice or index array

        """
        return self._idx

    def _idx_arr_to_slice(self, arr):
        """
        If the given index array can be represented as a slice, return a slice.
        Otherwise, return the index array.
        """
        if arr.size > 1:
            step = arr[1] - arr[0]
            for i in range(2, len(arr)):
                nstep = arr[i] - arr[i-1]
                if nstep != step:
                    return arr
            self._idx = slice(arr[0], arr[-1]+1, step)
        elif arr.size == 1:
            return slice(arr[0], arr[0]+1)
        else:
            return slice(0)


def to_slice(idxs):
    """Convert an index array to a slice if possible. Otherwise,
    return the index array.
    """
    if isinstance(idxs, ndarray):
        if idxs.size == 1:
            return slice(idxs[0], idxs[0]+1)
        elif idxs.size == 0:
            return slice(0)

        step = idxs[1]-idxs[0]

        if step <= 0:
            return idxs

        for i in range(2, idxs.size):
            if idxs[i] - idxs[i-1] != step:
                return idxs

        return slice(idxs[0], idxs[-1], step)
    else:
        raise RuntimeError("can't convert indices of type '%s' to a slice" %
                            str(type(idxs)))

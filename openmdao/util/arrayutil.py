
from six.moves import range
from numpy import ndarray
from itertools import product

def to_slice(idxs):
    """Convert an index array or list to a slice if possible. Otherwise,
    return the index array or list.
    """
    if isinstance(idxs, slice):
        return idxs
    elif isinstance(idxs, ndarray) or isinstance(idxs, list):
        if len(idxs) == 1:
            return slice(idxs[0], idxs[0]+1)
        elif len(idxs) == 0:
            return slice(0,0)

        if isinstance(idxs, ndarray):
            imin = idxs.min()
            imax = idxs.max()
        else:
            imin = min(idxs)
            imax = max(idxs)

        stride = idxs[1]-idxs[0]

        if stride == 0:
            return idxs

        for i in range(len(idxs)):
            if i and idxs[i] - idxs[i-1] != stride:
                return idxs

        if stride < 0:
            ## negative strides cause some failures, so just do positive for now
            #return slice(imax+1, imin, stride)
            return idxs
        else:
            return slice(imin, imax+1, stride)
    elif isinstance(idxs, int_types):
        return slice(idxs, idxs+1)
    else:
        raise RuntimeError("can't convert indices of type '%s' to a slice" %
                           str(type(idxs)))


def array_idx_iter(shape):
    """
    Return an iterator over the indices into a n-dimensional array.

    Parameters
    ----------
    shape : tuple
        shape of the array.
    """
    for p in product(*[range(s) for s in shape]):
        yield p



from six.moves import range
from numpy import ndarray, ones, zeros, cumsum
from itertools import product

#def to_slice(idxs):
    #"""Convert an index array or list to a slice if possible. Otherwise,
    #return the index array or list.
    #"""
    #if isinstance(idxs, slice):
        #return idxs
    #elif isinstance(idxs, ndarray) or isinstance(idxs, list):
        #if len(idxs) == 1:
            #return slice(idxs[0], idxs[0]+1)
        #elif len(idxs) == 0:
            #return slice(0,0)

        #if isinstance(idxs, ndarray):
            #imin = idxs.min()
            #imax = idxs.max()
        #else:
            #imin = min(idxs)
            #imax = max(idxs)

        #stride = idxs[1]-idxs[0]

        #if stride == 0:
            #return idxs

        #for i in range(len(idxs)):
            #if i and idxs[i] - idxs[i-1] != stride:
                #return idxs

        #if stride < 0:
            ### negative strides cause some failures, so just do positive for now
            ##return slice(imax+1, imin, stride)
            #return idxs
        #else:
            #return slice(imin, imax+1, stride)
    #elif isinstance(idxs, int_types):
        #return slice(idxs, idxs+1)
    #else:
        #raise RuntimeError("can't convert indices of type '%s' to a slice" %
                           #str(type(idxs)))


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


#def evenly_distrib_idxs(num_divisions, arr_size):
    #"""
    #Given a number of divisions and the size of an array, chop the array up
    #into pieces according to number of divisions, keeping the distribution
    #of entries as even as possible.

    #Args
    #----
    #num_divisions : int
        #Number of parts to divide the array into.

    #arr_size : int
        #Number of entries in the array.

    #Returns
    #-------
    #tuple
        #a tuple of (sizes, offsets), where sizes and offsets contain values for all
        #divisions.
    #"""
    #base = arr_size / num_divisions
    #leftover = arr_size % num_divisions
    #sizes = ones(num_divisions, dtype="int") * base

    ## evenly distribute the remainder across size-leftover procs,
    ## instead of giving the whole remainder to one proc
    #sizes[:leftover] += 1

    #offsets = zeros(num_divisions, dtype="int")
    #offsets[1:] = cumsum(sizes)[:-1]

    #return sizes, offsets

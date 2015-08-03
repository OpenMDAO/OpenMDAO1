""" Defines our differentiable data types. """

import numbers
import array
import numpy

real_types = [numbers.Real]
int_types = [numbers.Integral]
complex_types = [numbers.Complex]
iterable_types = [set, list, tuple, array.array]

real_types.extend([numpy.float32, numpy.float64])
int_types.extend([numpy.int32, numpy.int64])
complex_types.extend([numpy.complex])
iterable_types.append(numpy.ndarray)

# use these with isinstance to test for various types that include builtins
# and numpy types (if numpy is available)

complex_or_real_types = tuple(real_types+complex_types)
real_types = tuple(real_types)
int_types = tuple(int_types)
complex_types = tuple(complex_types)
iterable_types = tuple(iterable_types)


def is_differentiable(val):
    """ Determines if a quantity is differentiable based on its data type.

    Args
    ----
    val : object
        Some quantity from an OpenMDAO variable. Can be anything.

    Returns
    -------
    bool : Returns True if val is differentiable.
    """
    if isinstance(val, int_types):
        return False
    elif isinstance(val, real_types):
        return True
    elif isinstance(val, numpy.ndarray) and (str(val.dtype)[:5]=='float' or \
                                             str(val.dtype)[:7]=='complex'):
        return True
    return False

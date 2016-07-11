from collections import OrderedDict
from six import iteritems

def _jac_to_flat_dict(jac):
    """ Converts a double `dict` jacobian to a flat `dict` Jacobian. Keys go
    from [out][in] to [out,in].

    Args
    ----

    jac : dict of dicts of ndarrays
        Jacobian that comes from calc_gradient when the return_type is 'dict'.

    Returns
    -------

    dict of ndarrays
    """

    new_jac = OrderedDict()
    for key1, val1 in iteritems(jac):
        for key2, val2 in iteritems(val1):
            new_jac[(key1, key2)] = val2

    return new_jac



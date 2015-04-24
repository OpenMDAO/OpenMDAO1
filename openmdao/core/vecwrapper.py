from collections import OrderedDict
import numpy

from openmdao.util.types import is_differentiable

class VecWrapper(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.
    """
    def __init__(self):
        self.vec = None
        self._vardict = OrderedDict()

    def __getitem__(self, name):
        """Retrieve unflattened value of named var."""
        meta = self._vardict[name]
        shape = meta.get('shape')
        if shape is None:
            return meta['val']
        else:
            return meta['val'].reshape(shape)

    def __setitem__(self, name, value):
        """Set the value of the named var"""
        meta = self._vardict[name]
        if meta['size'] > 0:
            meta['val'][:] = value[:]
        else:
            meta['val'] = value

    def __len__(self):
        """Return the number of keys"""
        return len(self._vardict)

    def keys(self):
        return self._vardict.keys()

    def items(self):
        return self._vardict.items()


class SourceVecWrapper(VecWrapper):
    def __init__(self, unknowns, states, initialize=False):
        super(SourceVecWrapper, self).__init__()

        vec_size = 0
        for name, meta in unknowns.items():
            vec_size += self._add_var(name, meta, vec_size)

        for name, meta in states.items():
            vec_size += self._add_var(name, meta, vec_size, state=True)

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                meta['val'] = self.vec[meta['start']:meta['end']]

        if initialize:
            for name, meta in unknowns.items():
                self.vec[name] = meta['val']

            for name, meta in states.items():
                self.vec[name] = meta['val']


    def _add_var(self, name, meta, index, state=False):
        vmeta = self._vardict[name] = {}
        vmeta['state'] = state

        if 'shape' in meta:
            vmeta['shape'] = meta['shape']
            if 'val' in meta and not is_differentiable(val):
                var_size = 0
            else:
                var_size = 1
                for s in meta['shape']:
                    var_size *= s
        elif 'val' in meta:
            val = meta['val']
            if is_differentiable(val):
                if isinstance(val, numpy.ndarray):
                    var_size = val.size
                    # if they didn't specify the shape, get it here so we
                    # can unflatten the value we return from __getitem__
                    vmeta['shape'] = val.shape
                else:
                    var_size = 1
            else:
                var_size = 0
        else:
            raise ValueError("No value or shape given for '%s'" % name)

        vmeta['size'] = var_size
        if var_size > 0:
            vmeta['start'] = index
            vmeta['end'] = index + var_size

        return var_size

class TargetVecWrapper(VecWrapper):
    def __init__(self, params, srcvec):
        pass

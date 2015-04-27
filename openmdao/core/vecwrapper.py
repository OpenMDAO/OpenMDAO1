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
            if isinstance(value, numpy.ndarray):
                meta['val'][:] = value.flat[:]
            else:
                meta['val'][:] = value
        else:
            meta['val'] = value

    def __len__(self):
        """Return the number of keys"""
        return len(self._vardict)

    def keys(self):
        return self._vardict.keys()

    def items(self):
        return self._vardict.items()

    def metadata(self, name):
        return self._vardict[name]


class SourceVecWrapper(VecWrapper):
    def __init__(self, unknowns, states, store_noflats=False, parent=None):
        super(SourceVecWrapper, self).__init__()

        if parent is not None:
            self.create_view(unknowns, states, store_noflats, parent)
            return

        vec_size = 0
        for name, meta in unknowns.items():
            vmeta = self._add_var(name, meta, vec_size)
            if vmeta['size'] > 0 or store_noflats:
                self._vardict[name] = vmeta
                vec_size += vmeta['size']

        for name, meta in states.items():
            vmeta = self._add_var(name, meta, vec_size, state=True)
            if vmeta['size'] > 0 or store_noflats:
                self._vardict[name] = vmeta
                vec_size += vmeta['size']

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                meta['val'] = self.vec[meta['start']:meta['end']]

        # if store_noflats is True, this is the unknowns vecwrapper,
        # so initialize all of the values from the unknowns and states
        # dicts.
        if store_noflats:
            for name, meta in unknowns.items():
                self[name] = meta['val']

            for name, meta in states.items():
                self[name] = meta['val']

    def _add_var(self, name, meta, index, state=False):
        vmeta = {}
        vmeta['state'] = state

        if 'shape' in meta:
            shape = meta['shape']
            vmeta['shape'] = shape
            if 'val' in meta:
                val = meta['val']
                if not is_differentiable(val):
                    var_size = 0
                else:
                    if val.shape != shape:
                        raise ValueError("specified shape != val shape")
                    var_size = val.size
            else:
                # no val given, so assume they want a numpy float array
                meta['val'] = numpy.zeros(shape)
                var_size = meta['val'].size
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

        return vmeta

    def create_view(self, unknowns, states, store_noflats, parent):
        pass


class TargetVecWrapper(VecWrapper):
    def __init__(self, params, srcvec, initialize=True):
        super(TargetVecWrapper, self).__init__()
        vec_size = 0
        for name, meta in params.items():
            vec_size += self._add_var(name, meta, vec_size, srcvec)

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                meta['val'] = self.vec[meta['start']:meta['end']]
                self[name] = srcvec[name]

    def _add_var(self, name, meta, index, srcvec):
        vmeta = self._vardict[name] = {}

        srcval = srcvec[name]
        srcmeta = srcvec.metadata(name)
        var_size = srcmeta['size']

        vmeta['size'] = var_size
        if 'shape' in srcmeta:
            vmeta['shape'] = srcmeta['shape']

        if var_size > 0:
            vmeta['start'] = index
            vmeta['end'] = index + var_size
        else:
            vmeta['val'] = srcmeta['val']

        return var_size

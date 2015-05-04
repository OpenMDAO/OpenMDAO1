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
        self._slices = OrderedDict()

    def __getitem__(self, name):
        """Retrieve unflattened value of named var"""
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
        """Return the number of keys (variables)"""
        return len(self._vardict)

    def keys(self):
        """Return the keys (variable names)"""
        return self._vardict.keys()

    def items(self):
        return self._vardict.items()

    def metadata(self, name):
        return self._vardict[name]

    @staticmethod
    def create_source_vector(unknowns, store_noflats=False):
        """Create a vector storing a flattened array of the variables in unknowns.
        If store_noflats is True, then non-flattenable variables
        will also be stored.
        """

        self = VecWrapper()

        vec_size = 0
        for name, meta in unknowns.items():
            vmeta = self._add_source_var(name, meta, vec_size)
            var_size = vmeta['size']
            if var_size > 0 or store_noflats:
                if var_size > 0:
                    self._slices[meta['relative_name']] = (vec_size, vec_size + var_size)
                self._vardict[meta['relative_name']] = vmeta
                vec_size += var_size

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        # if store_noflats is True, this is the unknowns vecwrapper,
        # so initialize all of the values from the unknowns
        # dicts.
        if store_noflats:
            for name, meta in unknowns.items():
                self[meta['relative_name']] = meta['val']

        return self

    def _add_source_var(self, name, meta, index, state=False):
        """Add a variable to the vector. If the variable is differentiable,
        then allocate a range in the vector array to store it. Store the
        shape of the variable so it can be un-flattened later."""

        vmeta = {}
        vmeta['state'] = state
        vmeta['pathname'] = name

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

        return vmeta

    @staticmethod
    def create_target_vector(params, srcvec, my_params, connections, store_noflats=False):
        """Create a vector storing a flattened array of the variables in params.
        Variable shape and value are retrieved from srcvec
        """
        self = VecWrapper()

        vec_size = 0
        for pathname, meta in params.items():
            if pathname in my_params:
                # if connected, get metadata from the source
                src_pathname = connections.get(pathname)
                if src_pathname is None:
                    raise RuntimeError("Parameter %s is not connected" % pathname)
                relative_name = get_relative_varname(src_pathname, srcvec)
                src_meta = srcvec.metadata(relative_name)

                #TODO: check for self-containment of src and param
                vec_size += self._add_target_var(meta, vec_size, src_meta, store_noflats)

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        return self

    def _add_target_var(self, meta, index, src_meta, store_noflats):
        """Add a variable to the vector. Allocate a range in the vector array
        and store the shape of the variable so it can be un-flattened later."""

        name = meta['relative_name']
        vmeta = self._vardict[name] = {}

        var_size = src_meta['size']

        vmeta['size'] = var_size
        if 'shape' in src_meta:
            vmeta['shape'] = src_meta['shape']

        if var_size > 0:
            self._slices[name] = (index, index + var_size)
        elif store_noflats:
            vmeta['val'] = src_meta['val']

        return var_size

    def get_view(self, varmap):
        view = VecWrapper()
        view_size = 0

        start = -1
        for name, meta in self._vardict.items():
            if name in varmap:
                view._vardict[varmap[name]] = self._vardict[name]
                if meta['size'] > 0:
                    pstart, pend = self._slices[name]
                    if start == -1:
                        start = pstart
                        end = pend
                    else:
                        assert pstart == end, \
                               "%s not contiguous in block containing %s" % \
                               (name, varmap.keys())
                    end = pend
                    view._slices[varmap[name]] = (view_size, view_size + meta['size'])
                    view_size += meta['size']

        view.vec = self.vec[start:end]

        return view

def get_relative_varname(pathname, var_dict):
    """Returns the absolute pathname for the given relative variable
    name in the variable dictionary"""
    for rel_name, meta in var_dict.items():
        if meta['pathname'] == pathname:
            return rel_name
    raise RuntimeError("Relative name not found for %s" % pathname)

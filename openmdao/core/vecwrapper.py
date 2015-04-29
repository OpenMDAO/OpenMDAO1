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
    def create_source_vector(outputs, states, store_noflats=False):
        """Create a vector storing a flattened array of the variables in outputs
        and states. If store_noflats is True, then non-flattenable variables
        will also be stored. If a parent vector is provided, then this vector
        will provide a view into the parent vector"""

        self = VecWrapper()

        vec_size = 0
        for name, meta in outputs.items():
            vmeta = self._add_source_var(name, meta, vec_size)
            var_size = vmeta['size']
            if var_size > 0 or store_noflats:
                if var_size > 0:
                    self._slices[name] = (vec_size, vec_size + var_size)
                self._vardict[name] = vmeta
                vec_size += var_size

        for name, meta in states.items():
            vmeta = self._add_source_var(name, meta, vec_size, state=True)
            var_size = vmeta['size']
            if var_size > 0 or store_noflats:
                if var_size > 0:
                    self._slices[name] = (vec_size, vec_size + var_size)
                self._vardict[name] = vmeta
                vec_size += var_size

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        # if store_noflats is True, this is the unknowns vecwrapper,
        # so initialize all of the values from the outputs and states
        # dicts.
        if store_noflats:
            for name, meta in outputs.items():
                self[name] = meta['val']

            for name, meta in states.items():
                self[name] = meta['val']

        return self

    def _add_source_var(self, name, meta, index, state=False):
        """Add a variable to the vector. If the variable is differentiable,
        then allocate a range in the vector array to store it. Store the
        shape of the variable so it can be un-flattened later."""

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

        return vmeta

    @staticmethod
    def create_target_vector(group, params, srcvec, store_noflats=False):
        """Create a vector storing a flattened array of the variables in params.
        Variable shape and value are retrieved from srcvec
        """

        self = VecWrapper()

        vec_size = 0
        for name, meta in params.items():
            powner = meta.get('owner')
            source = meta.get('_source_')
            if source is not None:
                src_meta = srcvec.metadata(source)
            else:
                src_meta = srcvec.metadata(name)
            #TODO: check for self-containment of src and param
            vec_size += self._add_target_var(name, meta, vec_size, src_meta, store_noflats)

        self.vec = numpy.zeros(vec_size)

        for name, meta in self._vardict.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        return self

    def _add_target_var(self, name, meta, index, src_meta, store_noflats):
        """Add a variable to the vector. Allocate a range in the vector array
        and store the shape of the variable so it can be un-flattened later."""
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

    def get_view(self, varmap, is_target=False):
        view = VecWrapper()
        view_size = 0

        start = -1
        for name, meta in self._vardict.items():
            if name in varmap:
                if is_target:
                    if '_source_' not in meta:
                        continue
                    if '_source_' in meta and meta['_source_'] not in self._vardict:
                        continue
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

        if is_target:
            view.vec = numpy.zeros(view_size)
        else:
            view.vec = self.vec[start:end]

        return view

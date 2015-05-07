from collections import OrderedDict
import numpy

from openmdao.util.types import is_differentiable, int_types

class VecWrapper(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.
    """

    # other impls (petsc, etc) will define this differently
    idx_arr_type = 'i'

    def __init__(self):
        self.vec = None
        self._vardict = OrderedDict()
        self._slices = OrderedDict()

    def __getitem__(self, name):
        """Retrieve unflattened value of named var"""
        meta = self._vardict[name][0]
        shape = meta.get('shape')
        if shape is None:
            return meta['val']
        else:
            return meta['val'].reshape(shape)

    def __setitem__(self, name, value):
        """Set the value of the named var"""
        meta = self._vardict[name][0]
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
        """ iterate over the first metadata for each variable """
        for name, metadata_entry in self._vardict.items():
            yield name, metadata_entry[0]

    def values(self):
        """ iterate over the first metadata for each variable """
        for  metadata_entry in self._vardict.values():
            yield metadata_entry[0]

    def metadata(self, name):
        return self._vardict[name]

    def get_idxs(self, name):
        """
        Returns
        -------
        ndarray
            Index array containing all indices (possibly distributed) for the named variable.
        """
        # TODO: add support for returning slice objects

        meta = self._vardict[name][0]
        if meta.get('noflat'):
            raise RuntimeError("No indices can be provided for %s" % name)

        start, end = self._slices[name]
        return self.make_idx_array(start, end)

    @staticmethod
    def create_source_vector(unknowns_dict, store_noflats=False):
        """Create a vector storing a flattened array of the variables in unknowns.
        If store_noflats is True, then non-flattenable variables
        will also be stored.
        """

        self = VecWrapper()

        vec_size = 0
        for name, meta in unknowns_dict.items():
            vmeta = self._add_source_var(name, meta, vec_size)
            var_size = vmeta['size']
            if var_size > 0 or store_noflats:
                if var_size > 0:
                    self._slices[meta['relative_name']] = (vec_size, vec_size + var_size)
                # for a target (unknown) vector, there may be multiple
                # variables with relative name
                self._vardict.setdefault(meta['relative_name'], []).append(vmeta)
                vec_size += var_size

        self.vec = numpy.zeros(vec_size)

        # get the metadata from the first set of metadata in the list
        # (there will actually be only one for source variables)

        # map slices to the array
        for name, meta in self.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        # if store_noflats is True, this is the unknowns vecwrapper,
        # so initialize all of the values from the unknowns dicts.
        if store_noflats:
            for name, meta in unknowns_dict.items():
                self[meta['relative_name']] = meta['val']

        return self

    def _add_source_var(self, name, meta, index, state=False):
        """Add a variable to the vector. If the variable is differentiable,
        then allocate a range in the vector array to store it. Store the
        shape of the variable so it can be un-flattened later."""

        vmeta = meta.copy()
        vmeta['state'] = state
        vmeta['pathname'] = name

        if 'shape' in meta:
            shape = meta['shape']
            vmeta['shape'] = shape
            if 'val' in meta:
                val = meta['val']
                if not is_differentiable(val):
                    var_size = 0
                    vmeta['noflat'] = True
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
                vmeta['noflat'] = True
        else:
            raise ValueError("No value or shape given for '%s'" % name)

        vmeta['size'] = var_size

        return vmeta

    @staticmethod
    def create_target_vector(params_dict, srcvec, my_params, connections, store_noflats=False):
        """Create a vector storing a flattened array of the variables in params.
        Variable shape and value are retrieved from srcvec
        """
        self = VecWrapper()

        vec_size = 0
        for pathname, meta in params_dict.items():
            if pathname in my_params:
                # if connected, get metadata from the source
                src_pathname = connections.get(pathname)
                if src_pathname is None:
                    raise RuntimeError("Parameter %s is not connected" % pathname)
                src_rel_name = srcvec.get_relative_varname(src_pathname)
                src_meta = srcvec.metadata(src_rel_name)

                #TODO: check for self-containment of src and param
                vmeta = self._add_target_var(meta, vec_size, src_meta[0], store_noflats)
                vmeta['pathname'] = pathname

                vec_size += vmeta['size']

                self._vardict.setdefault(meta['relative_name'], []).append(vmeta)

        self.vec = numpy.zeros(vec_size)

        # get the size/val metadata from the first set of metadata in the list
        # (there may be metadata for multiple source variables for a target)

        # map slices to the array
        for name, meta in self.items():
            if meta['size'] > 0:
                start, end = self._slices[name]
                meta['val'] = self.vec[start:end]

        return self

    def _add_target_var(self, meta, index, src_meta, store_noflats):
        """Add a variable to the vector. Allocate a range in the vector array
        and store the shape of the variable so it can be un-flattened later."""

        vmeta = meta.copy()

        var_size = src_meta['size']

        vmeta['size'] = var_size
        if 'shape' in src_meta:
            vmeta['shape'] = src_meta['shape']

        if var_size > 0:
            self._slices[meta['relative_name']] = (index, index + var_size)
        elif src_meta.get('noflat') and store_noflats:
            vmeta['val'] = src_meta['val']
            vmeta['noflat'] = True

        return vmeta

    def get_view(self, varmap):
        view = VecWrapper()
        view_size = 0

        start = -1
        for name, meta in self.items():
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

        if start == -1: # no items found
            view.vec = self.vec[0:0]
        else:
            view.vec = self.vec[start:end]

        return view

    def make_idx_array(self, start, end):
        """ Return an index vector of the right int type for
        parallel or serial computation.
        """
        return numpy.arange(start, end, dtype=self.idx_arr_type)

    def merge_idxs(self, src_idxs, dest_idxs):
        """Return source and destination index arrays, built up from
        smaller index arrays and combined in order of ascending source
        index (to allow us to convert src indices to a slice in some cases).
        """
        assert(len(src_idxs) == len(dest_idxs))

        # filter out any zero length idx array entries
        src_idxs = [i for i in src_idxs if len(i)]
        dest_idxs = [i for i in dest_idxs if len(i)]

        if len(src_idxs) == 0:
            return make_idx_array(0, 0), make_idx_array(0,0)

        src_tups = list(enumerate(src_idxs))

        src_sorted = sorted(src_tups, key=lambda x: x[1].min())

        new_src = [idxs for i, idxs in src_sorted]
        new_dest = [dest_idxs[i] for i,_ in src_sorted]

        return idx_merge(new_src), idx_merge(new_dest)

    def get_relative_varname(self, abs_name):
        """Returns the relative pathname for the given absolute variable
        pathname in the variable dictionary

        Parameters
        ----------
        abs_name : str
            Absolute pathname of a variable

        Returns
        -------
        rel_name : str
            Relative name mapped to the given absolute pathname
        """
        for rel_name, meta_list in self._vardict.items():
            for meta in meta_list:
                if meta['pathname'] == abs_name:
                    return rel_name
        raise RuntimeError("Relative name not found for %s" % abs_name)

    def get_states(self):
        """
        Returns
        -------
            A list of names of state variables.
        """
        return [n for n,meta in self.items() if meta.get('state')]

    def get_vecvars(self):
        """
        Returns
        -------
            A list of names of 'flattenable' variables.
        """
        return [n for n,meta in self.items() if not meta.get('noflat')]

    def get_noflats(self):
        """
        Returns
        -------
            A list of names of 'unflattenable' variables.
        """
        return [n for n,meta in self.items() if meta.get('noflat')]





def idx_merge(idxs):
    """Combines a mixed iterator of int and iterator indices into an
    array of int indices.
    """
    if len(idxs) > 0:
        idxs = [i for i in idxs if isinstance(i, int_types) or
                len(i)>0]
        if len(idxs) > 0:
            if isinstance(idxs[0], int_types):
                return idxs
            else:
                return numpy.concatenate(idxs)
    return idxs


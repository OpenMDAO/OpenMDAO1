from collections import OrderedDict

import numpy
from numpy.linalg import norm

from openmdao.util.types import is_differentiable, int_types

class _flat_dict(object):
    """This is here to allow the user to use vec.flat['foo'] syntax instead
    of vec.flat('foo').
    """
    def __init__(self, vardict):
        self._dict = vardict

    def __getitem__(self, name):
        meta = self._dict[name][0]
        if meta.get('noflat'):
            raise ValueError("'%s' is non-flattenable" % name)
        return self._dict[name][0]['val']

class _NoflatWrapper(object):
    """We have to wrap noflat values in these in order to have param vec entries
    that are shared between parents and children all shared the same object
    reference, which would not be true for an unwrapped value.
    """
    def __init__(self, val):
        self.val = val

class VecWrapper(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.

    Attributes
    ----------
    idx_arr_type : dtype
        string indicating how index arrays are to be represented
        (the value 'i' indicates an numpy integer array, other
        implementations (petsc, etc) will define this differently)
    """

    idx_arr_type = 'i'

    def __init__(self):
        self.vec = None
        self._vardict = OrderedDict()
        self._slices = OrderedDict()

        # add a flat attribute that will have access method consistent
        # with non-flat access  (__getitem__)
        self.flat = _flat_dict(self._vardict)

    def _get_metadata(self, name):
        try:
            return self._vardict[name][0]
        except KeyError as error:
            msg  = "Variable '{name}' does not exist".format(name=name)
            raise KeyError(msg)

    def __getitem__(self, name):
        """Retrieve unflattened value of named var

        Parameters
        ----------
        name : str
            name of variable to get the value for

        Returns
        -------
            the unflattened value of the named variable
        """
        meta = self._get_metadata(name)

        if meta.get('noflat'):
            return meta['val'].val
        else:
            # if it doesn't have a shape, it's a float
            shape = meta.get('shape')
            if shape is None:
                return meta['val'][0]
            else:
                return meta['val'].reshape(shape)

    def __setitem__(self, name, value):
        """Set the value of the named variable

        Parameters
        ----------
        name : str
            name of variable to get the value for

        value :
            the unflattened value of the named variable
        """
        meta = self._get_metadata(name)

        if meta.get('noflat'):
            meta['val'].val = value
        else:
            if isinstance(value, numpy.ndarray):
                meta['val'][:] = value.flat[:]
            else:
                meta['val'][:] = value

    def __len__(self):
        """
        Returns
        -------
            the number of keys (variables) in this vector
        """
        return len(self._vardict)

    def keys(self):
        """
        Returns
        -------
            the keys (variable names) in this vector
        """
        return self._vardict.keys()

    def items(self):
        """Iterate over the first metadata for each variable

        Returns
        -------
            iterator
                iterator over the first metadata for each variable
        """
        for name, metadata_entry in self._vardict.items():
            yield name, metadata_entry[0]

    def values(self):
        """Iterate over the first metadata for each variable

        Returns
        -------
            iterator
                iterator over the first metadata for each variable
        """
        for metadata_entry in self._vardict.values():
            yield metadata_entry[0]

    def metadata(self, name):
        """Returns the metadata for the named variable. A target variable may
        have multiple sets of metadata due to having connections to multiple
        source variables, therefore a list of metadata dictionaries is returned.

        Parameters
        ----------
        name : str
            name of variable to get the metadata for

        Returns
        -------
            list of dict
                a list of the metadata dictionaries for the named variable
        """
        return self._vardict[name]

    def get_idxs(self, name):
        """Returns all of the indices for the named variable in this vector

        Parameters
        ----------
        name : str
            name of variable to get the indices for

        Returns
        -------
        ndarray
            Index array containing all indices (possibly distributed) for the named variable.
        """
        # TODO: add support for returning slice objects

        meta = self._vardict[name][0]
        if meta.get('noflat'):
            raise RuntimeError("No vector indices can be provided for non-flattenable variable '%s'" % name)

        start, end = self._slices[name]
        return self.make_idx_array(start, end)

    def setup_source_vector(self, unknowns_dict, store_noflats=False):
        """Configure this vector to store a flattened array of the variables
        in unknowns. If store_noflats is True, then non-flattenable variables
        will also be stored.

        Parameters
        ----------
        unknowns_dict : dict
            dictionary of metadata for unknown variables

        store_noflats : bool (optional)
            if True, then store non-flattenable (non-differentiable) variables
            by default only flattenable variables will be stired
        """
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

    def _add_source_var(self, name, meta, index):
        """Add a variable to the vector. If the variable is differentiable,
        then allocate a range in the vector array to store it. Store the
        shape of the variable so it can be un-flattened later.

        Parameters
        ----------
        name : str
            the name of the variable to add

        meta : dict
            metadata for the variable

        index : int
            index into the array where the variable value is to be stored
            (if flattenable)
        """

        vmeta = meta.copy()
        vmeta['pathname'] = name

        if 'shape' in meta:
            shape = meta['shape']
            vmeta['shape'] = shape
            if 'val' in meta:
                val = meta['val']
                if not is_differentiable(val):
                    var_size = 0
                    vmeta['noflat'] = True
                    vmeta['val'] = _NoflatWrapper(val)
                else:
                    if val.shape != shape:
                        raise ValueError("The specified shape of variable '%s' does not match the shape of its value." %
                                         name)
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
                vmeta['val'] = _NoflatWrapper(val)
        else:
            raise ValueError("No value or shape given for variable '%s'" % name)

        vmeta['size'] = var_size

        return vmeta

    def norm(self):
        """ Calculates the norm of this vector.

        Returns
        -------
        float
            Norm of the flattenable values in this vector.
        """
        return norm(self.vec)

    def setup_target_vector(self, parent_params_vec, params_dict, srcvec, my_params,
                            connections, store_noflats=False):
        """Configure this vector to store a flattened array of the variables
        in params. Variable shape and value are retrieved from srcvec.

        Parameters
        ----------
        parent_params_vec : `VecWrapper` or None
            `VecWrapper` of parameters from the parent `System`

        params_dict : `OrderedDict`
            Dictionary of parameter absolute name mapped to metadata dict

        srcvec : `VecWrapper`
            Source `VecWrapper` corresponding to the target `VecWrapper` we're building.

        my_params : list of str
            A list of absolute names of parameters that the `VecWrapper` we're building
            will 'own'.

        connections : dict of str : str
            A dict of absolute target names mapped to the absolute name of their
            source variable.

        store_noflats : bool (optional)
            If True, store unflattenable variables in the `VecWrapper` we're building.
        """
        vec_size = 0
        missing = []  # names of our params that we don't 'own'
        for pathname, meta in params_dict.items():
            if pathname in my_params:
                # if connected, get metadata from the source
                src_pathname = connections.get(pathname)
                if src_pathname is None:
                    raise RuntimeError("Parameter '%s' is not connected" % pathname)
                src_rel_name = srcvec.get_relative_varname(src_pathname)
                src_meta = srcvec.metadata(src_rel_name)

                vmeta = self._add_target_var(meta, vec_size, src_meta[0], store_noflats)
                vmeta['pathname'] = pathname

                vec_size += vmeta['size']

                self._vardict.setdefault(meta['relative_name'], []).append(vmeta)
            else:
                if parent_params_vec is not None:
                    missing.append(pathname)

        self.vec = numpy.zeros(vec_size)

        # get the size/val metadata from the first set of metadata in the list
        # (there may be metadata for multiple source variables for a target)

        # map slices to the array
        for name, metas in self._vardict.items():
            for meta in metas:
                if meta['size'] > 0:
                    start, end = self._slices[name]
                    meta['val'] = self.vec[start:end]

        # fill entries for missing params with views from the parent
        for pathname in missing:
            meta = params_dict[pathname]
            prelname = parent_params_vec.get_relative_varname(pathname)
            newmetas = parent_params_vec._vardict[prelname]
            for newmeta in newmetas:
                if newmeta['pathname'] == pathname:
                    newmeta = newmeta.copy()
                    newmeta['relative_name'] = meta['relative_name']
                    newmeta['owned'] = False # mark this param as not 'owned' by this VW
                    self._vardict.setdefault(meta['relative_name'],
                                             []).append(newmeta)

    def _add_target_var(self, meta, index, src_meta, store_noflats):
        """Add a variable to the vector. Allocate a range in the vector array
        and store the shape of the variable so it can be un-flattened later.

        Parameters
        ----------
        meta : dict
            metadata for the variable

        index : int
            index into the array where the variable value is to be stored
            (if flattenable)

        src_meta : dict
            metadata for the source variable that this target variable is
            connected to

        store_noflats : bool (optional)
            If True, store unflattenable variables in the `VecWrapper` we're building.
        """

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
        """Return a new `VecWrapper` that is a view into this one

        Parameters
        ----------
        varmap : dict
            mapping of variable names in the old `VecWrapper` to the names
            they will have in the new `VecWrapper`

        Returns
        -------
        `VecWrapper`
            a new `VecWrapper` that is a view into this one
        """
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

        Parameters
        ----------
        start : int
            the starting index

        end : int
            the ending index
        """
        return numpy.arange(start, end, dtype=self.idx_arr_type)

    def merge_idxs(self, src_idxs, tgt_idxs):
        """Return source and target index arrays, built up from
        smaller index arrays and combined in order of ascending source
        index (to allow us to convert src indices to a slice in some cases).

        Parameters
        ----------
        src_idxs : array
            source indices

        tgt_idxs : array
            target indices

        Returns
        -------
        array
            the merged index arrays
        """
        assert(len(src_idxs) == len(tgt_idxs))

        # filter out any zero length idx array entries
        src_idxs = [i for i in src_idxs if len(i)]
        tgt_idxs = [i for i in tgt_idxs if len(i)]

        if len(src_idxs) == 0:
            return self.make_idx_array(0, 0), self.make_idx_array(0,0)

        src_tups = list(enumerate(src_idxs))

        src_sorted = sorted(src_tups, key=lambda x: x[1].min())

        new_src = [idxs for i, idxs in src_sorted]
        new_tgt = [tgt_idxs[i] for i,_ in src_sorted]

        return idx_merge(new_src), idx_merge(new_tgt)

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
        raise RuntimeError("Relative name not found for variable '%s'" % abs_name)

    def get_states(self):
        """
        Returns
        -------
        list
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
        list
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

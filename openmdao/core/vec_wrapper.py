""" Class definition for VecWrapper"""

import sys
import numpy
from numpy import real, imag
from numpy.linalg import norm
from six import iteritems, itervalues, iterkeys
from six.moves import cStringIO

from collections import OrderedDict, namedtuple
from openmdao.core.fileref import FileRef
from openmdao.util.type_util import is_differentiable
from openmdao.util.string_util import get_common_ancestor

class _ByObjWrapper(object):
    """
    We have to wrap byobj values in these in order to have param vec entries
    that are shared between parents and children all share the same object
    reference, so that when the internal val attribute is changed, all
    `VecWrapper`s that contain a reference to the wrapper will see the updated
    value.
    """
    __slots__ = ['val']
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)

# using a slotted object here to save memory
class Accessor(object):
    __slots__ = ['val', 'imag_val', 'slice', 'meta', 'owned', 'pbo', 'remote',
                 'get', 'set', 'flat', 'probdata', 'vectype', 'disable_scale']
    def __init__(self, vecwrapper, slice, val, meta, probdata, alloc_complex,
                 owned=True, imag_val=None, dangling=False):
        """ Initialize this accessor.

        Args
        ----
        vecwrapper : `VecWrapper`
            `VecWrapper` that owns this `Accessor`.

        slice : `Slice` object
            A slice into the vector for this variable.

        val : float or ndarray
            Initial value of variable for this accessor.

        meta : dict
            Metadata for the variable collected from components.

        probdata : _ProbData
            A data object for Problem level data that we need in order to store
            flags that span multiple layers in the hierarchy.

        alloc_complex : bool, optional
            If True, allocate space for the imaginary part of the vector and
            configure all functions to support complex computation.

        owned : bool, optional
            True if this parameter is owned by the vecwrapper.

        imag_val : float or ndarray, optional
            Ininitial value for imaginary part of this vector. Only used under
            complex step, and always zero valued.

        dangling : bool, optional
            If True, this variable is an unconnected param.
        """
        self.owned = owned
        self.vectype = None
        self.disable_scale = False

        self.pbo = bool(dangling or meta.get('pass_by_obj'))
        self.remote = meta.get('remote')

        if alloc_complex:
            self.probdata = probdata

        if self.pbo and not isinstance(val, _ByObjWrapper):
            self.val = _ByObjWrapper(val)
        else:
            self.val = val
            if alloc_complex is True and not isinstance(val, _ByObjWrapper):
                if imag_val is None:
                    imag_val = val*0.0
                self.imag_val = imag_val

        if self.remote or self.pbo:
            self.slice = None
        else:
            self.slice = slice
        self.meta = meta

        self.get, self.flat = self._setup_get_funct(vecwrapper, meta, alloc_complex)
        self.set = self._setup_set_funct(vecwrapper, meta, alloc_complex)

    def _setup_get_funct(self, vecwrapper, meta, alloc_complex):
        """
        Returns a tuple of efficient closures (nonflat and flat) to access
        the value contained in the metadata.
        """

        val = meta['val']
        flatfunc = None
        self.vectype = vecwrapper.vectype

        if self.remote:
            return self._remote_access_error, self._remote_access_error

        scale, offset = meta.get('unit_conv', (None, None))

        # Pass by Object methods
        if self.pbo:
            if scale:
                return self._get_pbo_units, flatfunc
            else:
                return self._get_pbo, flatfunc

        shape = meta['shape']
        if vecwrapper.deriv_units:
            offset = 0.0
        is_scalar = shape == 1
        if is_scalar:
            shapes_same = True
        else:
            shapes_same = (shape == val.size or shape == (val.size,))

        # No unit conversion.
        # dparams vector does no unit conversion.
        if scale is None or vecwrapper.deriv_units:

            scaler = None
            # Need the unknowns vector to unscale the value when any get is
            # called during apply_linear, apply_nonlinear, or linearize. No
            # other vectors return scaled values on get.
            if vecwrapper.vectype == 'u':
                scaler = meta.get('scaler')

            if scaler:
                if alloc_complex:
                    flatfunc = self._get_arr_complex_scaler
                    if is_scalar:
                        func = self._get_scalar_complex_scaler
                    elif shapes_same:
                        func = flatfunc
                    else:
                        func = self._get_arr_diff_shape_complex_scaler
                else:
                    flatfunc = self._get_arr_scaler
                    if is_scalar:
                        func = self._get_scalar_scaler
                    elif shapes_same:
                        func = flatfunc
                    else:
                        func = self._get_arr_diff_shape_scaler

            else:
                if alloc_complex:
                    flatfunc = self._get_arr_complex
                    if is_scalar:
                        func = self._get_scalar_complex
                    elif shapes_same:
                        func = flatfunc
                    else:
                        func = self._get_arr_diff_shape_complex
                else:
                    flatfunc = self._get_arr
                    if is_scalar:
                        func = self._get_scalar
                    elif shapes_same:
                        func = flatfunc
                    else:
                        func = self._get_arr_diff_shape

        # We have a unit conversion
        else:
            if alloc_complex:
                flatfunc = self._get_arr_units_complex
                if is_scalar:
                    func = self._get_scalar_units_complex
                elif shapes_same:
                    func = flatfunc
                else:
                    func = self._get_arr_units_diff_shape_complex
            else:
                flatfunc = self._get_arr_units
                if is_scalar:
                    func = self._get_scalar_units
                elif shapes_same:
                    func = flatfunc
                else:
                    func = self._get_arr_units_diff_shape

        return func, flatfunc

    def _setup_set_funct(self, vecwrapper, meta, alloc_complex):
        """ Sets up our fast set functions."""

        if self.remote:
            return self._remote_access_error
        elif self.pbo:
            return self._set_pbo

        if meta['shape'] == 1:
            if alloc_complex:
                return self._set_scalar_complex
            else:
                return self._set_scalar
        else:
            if alloc_complex:
                return self._set_arr_complex
            else:
                return self._set_arr

    # accessor functions
    def _get_pbo(self):
        """pass by obj"""
        return self.val.val

    def _get_pbo_units(self):
        """Special unit conversions for pass by obj"""
        scale, offset = self.meta['unit_conv']
        vec = self.val.val + offset
        vec *= scale
        return vec

    def _get_arr(self):
        """Array with same shape."""
        return self.val

    def _get_arr_scaler(self):
        """Array with same shape, with scaling."""
        if self.disable_scale:
            return self.val
        if self.vectype == 'dr':
            scaler = 1.0/self.meta['scaler']
        else:
            scaler = self.meta['scaler']
        return scaler*self.val

    def _get_arr_complex(self):
        """Array with same shape, complex support."""
        if self.probdata.in_complex_step:
            return self.val + self.imag_val*1j
        else:
            return self.val

    def _get_arr_complex_scaler(self):
        """Array with same shape, complex support, with scaling."""
        scaler = self.meta['scaler']
        if self.probdata.in_complex_step:
            if self.disable_scale:
                return self.val + self.imag_val*1j
            return scaler*(self.val + self.imag_val*1j)
        else:
            if self.disable_scale:
                return self.val
            return scaler*self.val

    def _get_arr_diff_shape(self):
        """Array with different shape."""
        return self.val.reshape(self.meta['shape'])

    def _get_arr_diff_shape_scaler(self):
        """Array with different shape, with scaling."""
        if self.disable_scale:
            return self.val.reshape(self.meta['shape'])
        scaler = self.meta['scaler']
        return scaler*(self.val.reshape(self.meta['shape']))

    def _get_arr_diff_shape_complex(self):
        """Array with different shape, complex support."""
        if self.probdata.in_complex_step:
            val = self.val + self.imag_val*1j
        else:
            val = self.val
        return val.reshape(self.meta['shape'])

    def _get_arr_diff_shape_complex_scaler(self):
        """Array with different shape, complex support, with scaling."""
        if self.disable_scale:
            return val.reshape(self.meta['shape'])
        scaler = self.meta['scaler']
        if self.probdata.in_complex_step:
            val = self.val + self.imag_val*1j
        else:
            val = self.val
        return scaler*(val.reshape(self.meta['shape']))

    def _get_scalar(self):
        """Fast scalar."""
        return self.val[0]

    def _get_scalar_scaler(self):
        """Fast scalar, with scaling."""
        if self.disable_scale:
            return self.val[0]
        if self.vectype == 'dr':
            scaler = 1.0/self.meta['scaler']
        else:
            scaler = self.meta['scaler']
        return scaler*self.val[0]

    def _get_scalar_complex(self):
        """Fast scalar, complex support."""
        if self.probdata.in_complex_step:
            return self.val[0] + self.imag_val[0]*1j
        else:
            return self.val[0]

    def _get_scalar_complex_scaler(self):
        """Fast scalar, complex support, with scaling."""
        scaler = self.meta['scaler']
        if self.probdata.in_complex_step:
            if self.disable_scale:
                return self.val[0] + self.imag_val[0]*1j
            return scaler*(self.val[0] + self.imag_val[0]*1j)
        else:
            if self.disable_scale:
                return self.val[0]
            return scaler*self.val[0]

    def _get_arr_units(self):
        """Array with same shape and unit conversion."""
        scale, offset = self.meta['unit_conv']
        vec = self.val + offset
        vec *= scale
        return vec

    def _get_arr_units_complex(self):
        """Array with same shape and unit conversion, complex support."""
        if self.probdata.in_complex_step:
            val = self.val + self.imag_val*1j
        else:
            val = self.val
        scale, offset = self.meta['unit_conv']
        vec = val + offset
        vec *= scale
        return vec

    def _get_arr_units_diff_shape(self):
        """Array with diff shape and unit conversion."""
        scale, offset = self.meta['unit_conv']
        vec = self.val + offset
        vec *= scale
        return vec.reshape(self.meta['shape'])

    def _get_arr_units_diff_shape_complex(self):
        """Array with diff shape and unit conversion, complex support."""
        if self.probdata.in_complex_step:
            val = self.val + self.imag_val*1j
        else:
            val = self.val
        scale, offset = self.meta['unit_conv']
        vec = val + offset
        vec *= scale
        return vec.reshape(self.meta['shape'])

    def _get_scalar_units(self):
        """Scalar with unit conversion."""
        scale, offset = self.meta['unit_conv']
        return scale*(self.val[0] + offset)

    def _get_scalar_units_complex(self):
        """Scalar with unit conversion, complex support."""
        if self.probdata.in_complex_step:
            val = self.val[0] + self.imag_val[0]*1j
        else:
            val = self.val[0]
        scale, offset = self.meta['unit_conv']
        return scale*(val + offset)

    def _set_arr(self, value):
        """Set an array value."""
        self.val[:] = value.flat

    def _set_arr_complex(self, value):
        """Set an array value, complex support."""
        if self.probdata.in_complex_step:
            self.val[:] = real(value.flat)
            self.imag_val[:] = imag(value.flat)
        else:
            self.val[:] = value.flat

    def _set_scalar(self, value):
        """Set a scalar value."""
        self.val[0] = value

    def _set_scalar_complex(self, value):
        """Set a scalar value, complex support."""
        if self.probdata.in_complex_step:
            self.val[0] = value.real
            self.imag_val[0] = imag(value)
        else:
            self.val[0] = value

    def _set_pbo(self, value):
        self.val.val = value

    def _remote_access_error(self, value=None):
        msg = "Cannot access remote Variable '{name}' in this process."
        raise RuntimeError(msg.format(name=self.meta['pathname']))

class VecWrapper(object):
    """
    A dict-like container of a collection of variables.

    Args
    ----
    sysdata : _SysData
        A data object for system level data

    probdata : _ProbData
        A data object for Problem level data that we need in order to store
        flags that span multiple layers in the hierarchy.

    comm : an MPI communicator (real or fake)
        a communicator that can be used for distributed operations
        when running under MPI.  If not running under MPI, it is
        ignored

    Attributes
    ----------
    idx_arr_type : dtype, optional
        A dtype indicating how index arrays are to be represented.
        The value 'i' indicates an numpy integer array, other
        implementations, e.g., petsc, will define this differently.
    """

    idx_arr_type = 'i'

    def __init__(self, sysdata, probdata, comm=None):
        self.comm = comm
        self.vec = None
        self._dat = OrderedDict()

        # Automatic unit conversion in target vectors
        self.deriv_units = False

        # Scaling support in source vectors
        self.vectype = None

        # Supports complex step
        self.alloc_complex = False

        self._sysdata = sysdata
        self._probdata = probdata

        self.scale_cache = None
        self.units_cache = None

    def _flat(self, name):
        """
        Return a flat version of the named variable, including any necessary conversions.
        """
        return self._dat[name].flat()

    def metadata(self, name):
        """
        Returns the metadata for the named variable.

        Args
        ----
        name : str
            Name of variable to get the metadata for.

        Returns
        -------
        dict
            The metadata dict for the named variable.

        Raises
        -------
        KeyError
            If the named variable is not in this vector.
        """
        try:
            return self._dat[name].meta
        except KeyError as error:
            raise KeyError("Variable '%s' does not exist" % name)

    def __getitem__(self, name):
        """
        Retrieve unflattened value of named var.

        Args
        ----
        name : str
            Name of variable to get the value for.

        Returns
        -------
            The unflattened value of the named variable.
        """
        return self._dat[name].get()

    def __setitem__(self, name, value):
        """
        Set the value of the named variable.

        Args
        ----
        name : str
            Name of variable to get the value for.

        value :
            The unflattened value of the named variable.
        """
        self._dat[name].set(value)

    def __len__(self):
        """
        Returns
        -------
            The number of keys (variables) in this vector.
        """
        return len(self._dat)

    def __contains__(self, key):
        """
        Returns
        -------
            A boolean indicating if the given key (variable name) is in this vector.
        """

        return key in self._dat

    def __iter__(self):
        """
        Returns
        -------
            A dictionary iterator over the items in _dat.
        """
        return self._dat.__iter__()

    def vec_val_iter(self):
        """
        Returns
        -------
            An iterator over names and values of all variables found in the
            flattened vector, i.e., no pass_by_obj variables.
        """
        return ((n, acc.val) for n, acc in iteritems(self._dat)
                       if not acc.pbo)

    def keys(self):
        """
        Returns
        -------
        list or KeyView (python 3)
            the keys (variable names) in this vector.
        """
        return self._dat.keys()

    def iterkeys(self):
        """
        Returns
        -------
        iter of str
            the keys (variable names) in this vector.
        """
        return iterkeys(self._dat)

    def items(self):
        """
        Returns
        -------
        list of (str, dict)
            List of tuples containing the name and metadata dict for each
            variable.
        """
        return [(name, acc.meta) for name, acc in iteritems(self._dat)]

    def iteritems(self):
        """
        Returns
        -------
        iterator
            Iterator returning the name and metadata dict for each variable.
        """
        return ((name, acc.meta) for name, acc in iteritems(self._dat))

    def values(self):
        """
        Returns
        -------
        list of dict
            List containing metadata dict for each variable.
        """
        return [acc.meta for acc in itervalues(self._dat)]

    def itervalues(self):
        """
        Returns
        -------
        iter of dict
            Iterator yielding metadata dict for each variable.
        """
        return (acc.meta for acc in itervalues(self._dat))

    def _get_local_idxs(self, name, idx_dict, get_slice=False):
        """
        Returns all of the indices for the named variable in this vector.

        Args
        ----
        name : str
            Name of variable to get the indices for.

        get_slice : bool, optional
            If True, return the idxs as a slice object, if possible.

        Returns
        -------
        size
            The size of the named variable.

        ndarray
            Index array containing all local indices for the named variable.
        """
        try:
            slc = self._dat[name].slice
            if slc is None:
                return self.make_idx_array(0, 0)
        except KeyError:
            # this happens if 'name' doesn't exist in this process
            return self.make_idx_array(0, 0)

        start, end = slc

        if name in idx_dict:
            #TODO: possible slice conversion
            idxs = self.to_idx_array(idx_dict[name]) + start
            if idxs.size > (end-start) or max(idxs) >= end:
                raise RuntimeError("Indices of interest specified for '%s'"
                                   "are too large" % name)
            return idxs
        else:
            if get_slice:
                return slice(start, end)
            return self.make_idx_array(start, end)

    def norm(self):
        """
        Calculates the norm of this vector.

        Returns
        -------
        float
            Norm of our internal vector.
        """
        return norm(self.vec)

    def get_view(self, system, comm, varmap):
        """
        Return a new `VecWrapper` that is a view into this one.

        Args
        ----
        system : `System`
            System for which the view is being created.

        comm : an MPI communicator (real or fake)
            A communicator that is used in the creation of the view.

        varmap : dict
            Mapping of variable names in the old `VecWrapper` to the names
            they will have in the new `VecWrapper`.

        Returns
        -------
        `VecWrapper`
            A new `VecWrapper` that is a view into this one.
        """
        view = self.__class__(system._sysdata, system._probdata, comm)
        view.alloc_complex = self.alloc_complex
        view.vectype = self.vectype
        view_size = 0

        start = -1

        alloc_complex = self.alloc_complex

        # varmap is ordered, in the same order as _dat
        for name, pname in iteritems(varmap):
            if name in self._dat:
                acc = self._dat[name]
                if acc.pbo or acc.remote:
                    view._dat[pname] = Accessor(view, None, acc.val, acc.meta,
                                                self._probdata,
                                                alloc_complex)
                else:
                    pstart, pend = acc.slice
                    if start == -1:
                        start = pstart
                        end = pend
                    else:
                        assert pstart == end, \
                               "%s not contiguous in block containing %s" % \
                               (name, varmap.keys())
                    end = pend
                    meta = acc.meta

                    if alloc_complex:
                        imag_val = acc.imag_val
                    else:
                        imag_val = None

                    view._dat[pname] = Accessor(view,
                                                (view_size, view_size + meta['size']),
                                                acc.val, meta, self._probdata,
                                                alloc_complex, imag_val=imag_val)
                    view_size += meta['size']

        if start == -1: # no items found
            view.vec = self.vec[0:0]
            if alloc_complex:
                view.imag_vec = self.imag_vec[0:0]

        else:
            view.vec = self.vec[start:end]
            if alloc_complex:
                view.imag_vec = self.imag_vec[start:end]

        return view

    def make_idx_array(self, start, end):
        """
        Return an index vector of the right int type for
        the current implementation.

        Args
        ----
        start : int
            The starting index.

        end : int
            The ending index.

        Returns
        -------
        ndarray of idx_arr_type
            index array containing all indices from start up to but
            not including end
        """
        return numpy.arange(start, end, dtype=self.idx_arr_type)

    def to_idx_array(self, indices):
        """
        Given some iterator of indices, return an index array of the
        right int type for the current implementation.

        Args
        ----
        indices : iterator of ints
            An iterator of indices.

        Returns
        -------
        ndarray of idx_arr_type
            Index array containing all of the given indices.

        """
        return numpy.array(indices, dtype=self.idx_arr_type)

    def merge_idxs(self, idxs):
        """
        Return source and target index arrays, built up from
        smaller index arrays.

        Args
        ----
        idxs : array
            Indices.

        Returns
        -------
        ndarray of idx_arr_type
            Index array containing all of the merged indices.

        """
        if len(idxs) == 0:
            return self.make_idx_array(0, 0)

        return numpy.concatenate(idxs)

    def dump(self, out_stream=sys.stdout):  # pragma: no cover
        """
        Args
        ----
        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to return a str.
        """

        if out_stream is None:
            out_stream = cStringIO()
            return_str = True
        else:
            return_str = False

        lens = [len(n) for n in self.keys()]
        nwid = max(lens) if lens else 10
        vlens = [len(repr(self[v])) for v in self.keys()]
        vwid = max(vlens) if vlens else 1
        for acc in itervalues(self._dat):
            if acc.pbo:
                defwid = 8
                break
        else:
            defwid = 1

        slens = [len('[{0[0]}:{0[1]}]'.format(self._dat[v].slice))
                   for v in self.keys()
                       if self._dat[v].slice is not None]+[defwid]
        swid = max(slens)

        for v, acc in iteritems(self._dat):
            if acc.pbo or acc.remote:
                continue
            if self._dat[v].slice is not None:
                uslice = '[{0[0]}:{0[1]}]'.format(self._dat[v].slice)
            else:
                uslice = ''
            template = "{0:<{nwid}} {1:<{swid}} {2:>{vwid}}\n"
            out_stream.write(template.format(v,
                                             uslice,
                                             repr(self[v]),
                                             nwid=nwid,
                                             swid=swid,
                                             vwid=vwid))

        for v, acc in iteritems(self._dat):
            if acc.pbo and not acc.remote:
                template = "{0:<{nwid}} {1:<{swid}} {2}\n"
                out_stream.write(template.format(v, '(by obj)',
                                                 repr(self[v]),
                                                 nwid=nwid,
                                                 swid=swid))
        if return_str:
            return out_stream.getvalue()


class SrcVecWrapper(VecWrapper):
    """ Vecwrapper for unknowns, resids, dunknowns, and dresids."""

    def setup(self, unknowns_dict, relevance=None, var_of_interest=None,
              store_byobjs=False, shared_vec=None, alloc_complex=False,
              vectype='u'):
        """
        Configure this vector to store a flattened array of the variables
        in unknowns. If store_byobjs is True, then 'pass by object' variables
        will also be stored.

        Args
        ----
        unknowns_dict : dict
            Dictionary of metadata for unknown variables collected from
            components.

        relevance : `Relevance` object
            Object that knows what vars are relevant for each var_of_interest.

        var_of_interest : str or None
            Name of the current variable of interest.

        store_byobjs : bool, optional
            If True, then store 'pass by object' variables.
            By default only 'pass by vector' variables will be stored.

        shared_vec : ndarray, optional
            If not None, create vec as a subslice of this array.

        alloc_complex : bool, optional
            If True, allocate space for the imaginary part of the vector and
            configure all functions to support complex computation.

        vectype : str('u'), optional
            Type of vector, can be 'u' (unknown), 'r' (resids), 'du' (dunknowns),
            or 'dr' (dresids).
        """

        # dunknowns/dresids vector has some additional behavior
        self.vectype = vectype

        vec_size = 0
        to_prom_name = self._sysdata.to_prom_name

        for path, meta in iteritems(unknowns_dict):
            promname = to_prom_name[path]
            if relevance is None or relevance.is_relevant(var_of_interest,
                                                    meta['top_promoted_name']):
                if ('pass_by_obj' not in meta or not meta['pass_by_obj']) and \
                       ('remote' not in meta or not meta['remote']):
                    slc = (vec_size, vec_size + meta['size'])
                    vec_size += meta['size']
                    self._dat[promname] = Accessor(self, slc, meta['val'], meta,
                                                   self._probdata, alloc_complex)
                else:
                    self._dat[promname] = Accessor(self, None, meta['val'], meta,
                                                   self._probdata, alloc_complex)

        if shared_vec is not None:
            self.vec = shared_vec[:vec_size]
        else:
            self.alloc_complex = alloc_complex
            self.vec = numpy.zeros(vec_size)
            if alloc_complex:
                self.imag_vec = numpy.zeros(vec_size)

        # map slices to the array
        for name, acc in iteritems(self._dat):
            if not acc.pbo:
                if acc.remote:
                    acc.val = numpy.empty(0, dtype=float)
                    if alloc_complex:
                        acc.imag_val = numpy.empty(0, dtype=float)
                else:
                    start, end = acc.slice
                    acc.val = self.vec[start:end]
                    if alloc_complex:
                        acc.imag_val = self.imag_vec[start:end]
                    meta = acc.meta
                    if store_byobjs:
                        # if store_byobjs is True, this is the unknowns vecwrapper,
                        # so initialize all of the values from the unknowns dicts.
                        if meta['shape'] == 1:
                            acc.val[0] = meta['val']
                        else:
                            acc.val[:] = meta['val'].flat

    def _get_flattened_sizes(self):
        """
        Collect all sizes of vars stored in our internal vector.

        Returns
        -------
        list of lists of (name, size) tuples
            A one entry list containing a list of tuples mapping var name to
            local size for 'pass by vector' variables.
        """
        return [[(n, acc.meta['size']) for n, acc in iteritems(self._dat)
                        if not acc.pbo]]

    def distance_along_vector_to_limit(self, alpha, duvec):
        """ Returns a new alpha so that new_u = current_u + alpha*duvec does
        not violate any `lower` or `upper` limits if specified.

        Args
        -----
        alpha: float
            Initial value for step in gradient direction.
        duvec: `Vecwrapper`
            Direction to apply step. generally the gradient.

        Returns
        --------
        float
            New step size, backtracked to prevent violation."""

        # A single index of the gradient can be zero, so we want to suppress
        # the warnings from numpy.
        old_warn = numpy.geterr()
        numpy.seterr(divide='ignore')

        new_alpha = alpha
        for name, meta in iteritems(self):

            if 'remote' in meta:
                continue

            val = self[name]

            upper = meta.get('upper')
            if upper is not None:
                alpha_bound = numpy.min((upper - val)/duvec[name])
                if alpha_bound >= 0.0:
                    new_alpha = min(new_alpha, alpha_bound)

            lower = meta.get('lower')
            if lower is not None:
                alpha_bound = numpy.min((lower - val)/duvec[name])
                if alpha_bound >= 0.0:
                    new_alpha = min(new_alpha, alpha_bound)

        # Return numpy warn to what it was
        numpy.seterr(divide=old_warn['divide'])

        return max(0.0, new_alpha)

    def _scale_derivatives(self):
        """ Applies derivative of the scaling factor to the contents sitting
        in dunknowns or dresids.
        """
        if self.scale_cache is None:
            self._cache_scalers()

        for name, scaler, resid_scaler in self.scale_cache:
            acc = self._dat[name]

            total_scale = 1.0
            if resid_scaler:
                total_scale /= resid_scaler
            if scaler:
                total_scale *= scaler
            acc.val *= total_scale

    def _scale_values(self):
        """ Applies the 'scaler' or 'resid_scaler' to the quantities sitting
        in the unknown or residual vectors.
        """
        if self.scale_cache is None:
            self._cache_scalers()

        for name, scaler, resid_scaler in self.scale_cache:
            acc = self._dat[name]
            if scaler:
                scaler = 1.0/scaler
                acc.val *= scaler
                acc.disable_scale = False

            elif resid_scaler:
                resid_scaler = 1.0/resid_scaler
                acc.val *= resid_scaler
                acc.disable_scale = False

    def _disable_scaling(self):
        """ Turns off automatic scaling when getting a value via the
        dictionary accessor. It is only turned off in the unknowns vector
        during solve_nonlinear to allow the user to get a reference to the
        unknown so that it can be flled by index."""

        if self.scale_cache is None:
            self._cache_scalers()

        for name, scaler, _ in self.scale_cache:
            if scaler:
                acc = self._dat[name]
                acc.disable_scale = True

    def _cache_scalers(self):
        """ Caches the scalers so we don't have to do a lot of looping."""

        scale_cache = []
        for name, acc in iteritems(self._dat):
            meta = acc.meta
            scaler = meta.get('scaler')
            resid_scaler = meta.get('resid_scaler')

            if scaler and self.vectype == 'r':
                scaler = None

            if resid_scaler and self.vectype == 'u':
                resid_scaler = None

            if self.vectype == 'dr':
                if not meta.get('state') and scaler is not None:
                    scaler = 1.0/scaler
                else:
                    scaler = None

            elif self.vectype == 'du':
                resid_scaler = None
                if not meta.get('state'):
                    scaler = None

            if scaler or resid_scaler:
                scale_cache.append((name, scaler, resid_scaler))


        self.scale_cache = scale_cache

class TgtVecWrapper(VecWrapper):
    """ VecWrapper for params and dparams. """

    def setup(self, parent_params_vec, params_dict, srcvec, my_params,
              connections, relevance=None, var_of_interest=None,
              store_byobjs=False, shared_vec=None, alloc_complex=False):
        """
        Configure this vector to store a flattened array of the variables
        in params_dict. Variable shape and value are retrieved from srcvec.

        Args
        ----
        parent_params_vec : `VecWrapper` or None
            `VecWrapper` of parameters from the parent `System`.

        params_dict : `OrderedDict`
            Dictionary of parameter absolute name mapped to metadata dict.

        srcvec : `VecWrapper`
            Source `VecWrapper` corresponding to the target `VecWrapper` we're building.

        my_params : list of str
            A list of absolute names of parameters that the `VecWrapper` we're building
            will 'own'.

        connections : dict of str : str
            A dict of absolute target names mapped to the absolute name of their
            source variable.

        relevance : `Relevance` object
            Object that knows what vars are relevant for each var_of_interest.

        var_of_interest : str or None
            Name of the current variable of interest.

        store_byobjs : bool, optional
            If True, store 'pass by object' variables in the `VecWrapper` we're building.

        shared_vec : ndarray, optional
            If not None, create vec as a subslice of this array.

        alloc_complex : bool, optional
            If True, allocate space for the imaginary part of the vector and
            configure all functions to support complex computation.
        """
        # dparams vector has some additional behavior
        if not store_byobjs:
            self.deriv_units = True

        src_to_prom_name = srcvec._sysdata.to_prom_name
        scoped_name = self._sysdata._scoped_abs_name
        vec_size = 0
        missing = []  # names of our params that we don't 'own'
        syspath = self._sysdata.pathname + '.'

        for meta in itervalues(params_dict):
            if relevance is None or relevance.is_relevant(var_of_interest,
                                                          meta['top_promoted_name']):
                pathname = meta['pathname']
                if pathname in my_params:
                    # if connected, get metadata from the source
                    try:
                        src = connections[pathname]
                    except KeyError:
                        raise RuntimeError("Parameter '%s' is not connected" % pathname)
                    src_pathname, idxs = src
                    src_rel_name = src_to_prom_name[src_pathname]
                    src_acc = srcvec._dat[src_rel_name]

                    slc, val = self._setup_var_meta(pathname, meta, vec_size,
                                                    src_acc, store_byobjs)

                    if 'remote' not in meta or not meta['remote']:
                        vec_size += meta['size']

                    self._dat[scoped_name(pathname)] = Accessor(self, slc, val,
                                                                meta,
                                                                self._probdata,
                                                                alloc_complex)

                elif parent_params_vec is not None and pathname in connections:
                    src, _ = connections[pathname]
                    common = get_common_ancestor(src, pathname)
                    if (common == self._sysdata.pathname or
                                                syspath not in common):
                        missing.append(pathname)

        if shared_vec is not None:
            self.vec = shared_vec[:vec_size]
        else:
            self.alloc_complex = alloc_complex
            self.vec = numpy.zeros(vec_size)
            if alloc_complex:
                self.imag_vec = numpy.zeros(vec_size)

        # map slices to the array
        for acc in itervalues(self._dat):
            if not (acc.pbo or acc.remote):
                start, end = acc.slice
                acc.val = self.vec[start:end]
                if alloc_complex:
                    acc.imag_val = self.imag_vec[start:end]

        # fill entries for missing params with views from the parent
        if parent_params_vec is not None:
            parent_scoped_name = parent_params_vec._sysdata._scoped_abs_name

        for pathname in missing:
            parent_acc = parent_params_vec._dat[parent_scoped_name(pathname)]
            newmeta = parent_acc.meta
            if newmeta['pathname'] == pathname:

                if alloc_complex is True and not newmeta.get('pass_by_obj'):
                    imag_val = parent_acc.imag_val
                else:
                    imag_val = None

                # mark this param as not 'owned' by this VW
                self._dat[scoped_name(pathname)] = Accessor(self, None,
                                                            parent_acc.val,
                                                            newmeta, self._probdata,
                                                            alloc_complex,
                                                            owned=False,
                                                            imag_val=imag_val)

        if self.deriv_units:
            self._cache_units()


    def _setup_var_meta(self, pathname, meta, index, src_acc, store_byobjs):
        """
        Populate the metadata dict for the named variable.

        Args
        ----
        pathname : str
            Absolute name of the variable.

        meta : dict
            Metadata for the variable collected from components.

        index : int
            Index into the array where the variable value is to be stored
            (if variable is not 'pass by object').

        src_acc : Accessor
            Accessor object for the source variable that this target variable is
            connected to.

        store_byobjs : bool, optional
            If True, store 'pass by object' variables in the `VecWrapper`
            we're building.
        """
        src_meta = src_acc.meta

        val = meta['val']

        if 'src_indices' not in meta and 'src_indices' not in src_meta:
            meta['size'] = src_meta['size']

        if src_acc.pbo:
            if not meta.get('remote') and store_byobjs and not isinstance(val, FileRef):
                val = src_acc.val
            meta['pass_by_obj'] = True
            slc = None
        elif meta.get('remote'):
            slc = None
        else:
            slc = (index, index + meta['size'])

        return slc, val

    def _add_unconnected_var(self, pathname, meta):
        """
        Add an entry to this vecwrapper for the given unconnected variable so the
        component can access its value through the vecwrapper.
        """
        if 'val' in meta:
            val = meta['val']
        elif 'shape' in meta:
            shape = meta['shape']
            val = numpy.zeros(shape)
        else:
            raise RuntimeError("Unconnected param '%s' has no specified val or shape" %
                               pathname)

        self._dat[self._sysdata._scoped_abs_name(pathname)] = \
                 Accessor(self, None, val, meta, self._probdata, False,
                          dangling=True)

    def _get_flattened_sizes(self):
        """
        Returns
        -------
        list of lists of tuples of the form (name, size)
            A one entry list of lists with tuples pairing names to local sizes
            of owned, local params in this `VecWrapper`.
        """
        return [[(n, acc.meta['size']) for n, acc in iteritems(self._dat)
                        if acc.owned and not acc.pbo]]

    def _apply_unit_derivatives(self):
        """ Applies derivative of the unit conversion factor to params
        sitting in vector.
        """

        if self.deriv_units:
            for name, val in self.units_cache:
                self._dat[name].val *= val

    def _cache_units(self):
        """ Caches the scalers so we don't have to do a lot of looping."""

        units_cache = []
        for name, acc in iteritems(self._dat):
            meta = acc.meta
            if 'unit_conv' in meta:
                units_cache.append((name, meta['unit_conv'][0]))

        self.units_cache = units_cache


class _PlaceholderVecWrapper(object):
    """
    A placeholder for a dict-like container of a collection of variables.

    Args
    ----
    name : str
        the name of the vector
    """

    def __init__(self, name=''):
        self.name = name

    def __getitem__(self, name):
        """
        Retrieve unflattened value of named var. Since this is just a
        placeholder, will raise an exception stating that setup() has
        not been called yet.

        Args
        ----
        name : str
            Name of variable to get the value for.

        Raises
        ------
        AttributeError
        """
        raise AttributeError("'%s' has not been initialized, "
                             "setup() must be called before '%s' can be accessed" %
                             (self.name, name))

    def __contains__(self, name):
        self.__getitem__(name)

    def __setitem__(self, name, value):
        """
        Set the value of the named variable. Since this is just a
        placeholder, will raise an exception stating that setup() has
        not been called yet.

        Args
        ----
        name : str
            Name of variable to get the value for.

        value :
            The unflattened value of the named variable.

        Raises
        ------
        AttributeError
        """
        raise AttributeError("'%s' has not been initialized, "
                             "setup() must be called before '%s' can be accessed" %
                             (self.name, name))

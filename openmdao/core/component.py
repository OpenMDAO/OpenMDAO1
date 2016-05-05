""" Defines the base class for a Component in OpenMDAO."""
from __future__ import print_function

import sys
import os
import re
import warnings

from collections import OrderedDict
from itertools import chain
from six import iteritems, itervalues, iterkeys

import numpy as np

from openmdao.core.basic_impl import BasicImpl
from openmdao.core.system import System
from openmdao.core.mpi_wrap import MPI
from openmdao.core.vec_wrapper import _ByObjWrapper
from openmdao.core.vec_wrapper_complex_step import ComplexStepSrcVecWrapper, \
                                                   ComplexStepTgtVecWrapper
from openmdao.core.fileref import FileRef
from openmdao.util.type_util import is_differentiable

# Object to represent default value for `add_output`.
_NotSet = object()

# regex to check for valid variable names.
namecheck_rgx = re.compile(
    '([_a-zA-Z][_a-zA-Z0-9]*)+(\:[_a-zA-Z][_a-zA-Z0-9]*)*')

trace = os.environ.get('OPENMDAO_TRACE')
if trace:
    from openmdao.core.mpi_wrap import debug

empty_arr = np.zeros(0)


class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its params to produce unknowns, which can be
    explicit outputs or implicit states.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative
    fd_options['extra_check_partials_form'] :  None or str
        Finite difference mode: ("forward", "backward", "central", "complex_step")
        During check_partial_derivatives, you can optionally do a
        second finite difference with a different mode.
    fd_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.
    """

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup_vars = False
        self._jacobian_cache = {}

        self._init_params_dict = OrderedDict() # for storage of initial var data
        self._init_unknowns_dict = OrderedDict() # for storage of initial var data

        # keep a list of nondifferentiable vars without user set 'pass_by_obj'
        # metadata for use later in check_setup
        self._pbo_warns = []
        self._run_apply = False

    def _get_initial_val(self, val, shape):
        """ Determines initial value based on starting val and shape."""
        if val is _NotSet:
            # Interpret a shape of 1 to mean scalar.
            if shape == 1:
                return 0.
            return np.zeros(shape)
        return val

    def _check_val(self, name, var_type, val, shape):
        """ Raises and exception if the user doesn't specify the right info
        in val and shape."""
        if val is _NotSet and shape is None:
            msg = ("Shape of {var_type} '{name}' must be specified because "
                   "'val' is not set")
            msg = msg.format(var_type=var_type, name=name)
            raise ValueError(msg)

    def _add_variable(self, name, val, **kwargs):
        """ Contruct metadata for new variable.

        Args
        ----
        name : string
            Name of the variable.

        val : float or ndarray or object
            Initial value for the variable.

        **kwargs
            Arbitrary keyword arguments to be added to metadata.

        Raises
        ------
        RuntimeError
            If name is already in use or if setup has already been performed.

        NameError
            If name is not valid.

        ValueError
            If a valid value or shape is not specified.
        """
        shape = kwargs.get('shape')
        self._check_varname(name)
        meta = kwargs.copy()

        if isinstance(val, FileRef):
            val._set_meta(kwargs)

        meta['val'] = val = self._get_initial_val(val, shape)

        if is_differentiable(val) and not meta.get('pass_by_obj'):
            if isinstance(val, np.ndarray):
                meta['size'] = val.size
                meta['shape'] = val.shape
            else:
                meta['size'] = 1
                meta['shape'] = 1
        else:
            if not meta.get('pass_by_obj'):
                self._pbo_warns.append((name, val))

            meta['size'] = 0
            meta['pass_by_obj'] = True

        if isinstance(shape, int) and shape > 1:
            meta['shape'] = (shape,)

        return meta

    def add_param(self, name, val=_NotSet, **kwargs):
        """ Add a `param` input to this component.

        Args
        ----
        name : string
            Name of the input.

        val : float or ndarray or object
            Initial value for the input.
        """

        if 'resid_scaler' in kwargs:
            msg = ("resid_scaler is only supported for states.")
            raise ValueError(msg)

        if 'scaler' in kwargs:
            msg = ("scaler is only supported for outputs and states.")
            raise ValueError(msg)

        self._init_params_dict[name] = self._add_variable(name, val, **kwargs)

    def add_output(self, name, val=_NotSet, **kwargs):
        """ Add an output to this component.

        Args
        ----
        name : string
            Name of the variable output.

        val : float or ndarray or object
            Initial value for the output. While the value is overwritten during
            execution, it is useful for infering size.
        """

        if 'resid_scaler' in kwargs:
            msg = ("resid_scaler is only supported for states.")
            raise ValueError(msg)

        if 'scaler' in kwargs:
            scaler = kwargs['scaler']
            if scaler == 0:
                msg = ("scaler value must be nonzero.")
                raise ValueError(msg)

            kwargs['scaler'] = float(scaler)

        shape = kwargs.get('shape')
        self._check_val(name, 'output', val, shape)
        self._init_unknowns_dict[name] = self._add_variable(name, val, **kwargs)

    def add_state(self, name, val=_NotSet, **kwargs):
        """ Add an implicit state to this component.

        Args
        ----
        name : string
            Name of the state.

        val : float or ndarray
            Initial value for the state.
        """

        if 'scaler' in kwargs:
            scaler = kwargs['scaler']
            if scaler == 0:
                msg = ("scaler value must be nonzero.")
                raise ValueError(msg)

            kwargs['scaler'] = float(scaler)

        if 'resid_scaler' in kwargs:
            resid_scaler = kwargs['resid_scaler']
            if resid_scaler == 0:
                msg = ("resid_scaler value must be nonzero.")
                raise ValueError(msg)

            kwargs['resid_scaler'] = float(resid_scaler)

        shape = kwargs.get('shape')
        self._check_val(name, 'state', val, shape)
        args = self._add_variable(name, val, **kwargs)
        args['state'] = True
        self._init_unknowns_dict[name] = args
        self._run_apply = True

    def set_var_indices(self, name, val=_NotSet, shape=None,
                        src_indices=None):
        """ Sets the 'src_indices' metadata of an existing variable
        on this component, as well as its value, size, shape, and
        global size.

        This only works for numpy array variables.

        Args
        ----
        name : string
            Name of the variable.

        val : ndarray, optional
            Initial value for the variable.

        shape : tuple, optional
            Specifies the shape of the ndarray value

        src_indices : array of indices
            An index array indicating which entries in the distributed
            version of this variable are present in this process.

        """
        meta = self._init_params_dict.get(name)
        if meta is None:
            meta = self._init_unknowns_dict[name]

        if src_indices is None:
            raise ValueError("You must provide src_indices for variable '%s'" %
                             name)

        if not isinstance(meta['val'], np.ndarray):
            raise ValueError("resize_var() can only be called for numpy "
                             "array variables, but '%s' is of type %s" %
                             (name, type(meta['val'])))

        if val is _NotSet:
            if shape:
                val = np.zeros(shape, dtype=meta['val'].dtype)
            else:
                # assume value is a 1-D array
                val = np.zeros((len(src_indices),), dtype=meta['val'].dtype)

        if val.size != len(src_indices):
            raise ValueError("The size (%d) of the array '%s' doesn't match the "
                             "size (%d) of the specified indices." %
                             (val.size, name, len(src_indices)))
        meta['val'] = val
        meta['shape'] = val.shape
        meta['size'] = val.size
        meta['src_indices'] = src_indices

    def _check_varname(self, name):
        """ Verifies that a variable name is valid. Also checks for
        duplicates."""
        if self._post_setup_vars:
            raise RuntimeError("%s: can't add variable '%s' because setup has already been called." %
                               (self.pathname, name))
        if name in self._init_params_dict or name in self._init_unknowns_dict:
            raise RuntimeError("%s: variable '%s' already exists." %
                               (self.pathname, name))

        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError("%s: '%s' is not a valid variable name." %
                            (self.pathname, name))

    def setup_distrib(self):
        """
        Override this in your Component to set specific indices that will be
        pulled from source variables to fill your parameters.  This method
        should set the 'src_indices' metadata for any parameters or
        unknowns that require it.
        """
        pass

    def _get_fd_params(self):
        """
        Get the list of parameters that are needed to perform a
        finite difference on this `Component`.

        Returns
        -------
        list of str
            List of names of params for this `Component` .
        """
        if self._fd_params is None:
            self._fd_params = [k for k, acc in iteritems(self.params._dat)
                                   if not acc.pbo]
        return self._fd_params

    def _get_fd_unknowns(self):
        """
        Get the list of unknowns that are needed to perform a
        finite difference on this `Group`.

        Returns
        -------
        list of str
            List of names of unknowns for this `Component`.
        """
        return [k for k, acc in iteritems(self.unknowns._dat)
                      if not acc.pbo]

    def _setup_variables(self, compute_indices=False):
        """
        Returns copies of our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        Args
        ----

        compute_indices : bool, optional
            If True, call setup_distrib() to set values of
            'src_indices' metadata.

        """
        to_prom_name = self._sysdata.to_prom_name = {}
        to_abs_uname = self._sysdata.to_abs_uname = {}
        to_abs_pnames = self._sysdata.to_abs_pnames = OrderedDict()
        to_prom_uname = self._sysdata.to_prom_uname = OrderedDict()
        to_prom_pname = self._sysdata.to_prom_pname = OrderedDict()

        if MPI and compute_indices and self.is_active():
            if hasattr(self, 'setup_distrib_idxs'):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn("setup_distrib_idxs is deprecated, use setup_distrib instead.",
                              DeprecationWarning,stacklevel=2)
                warnings.simplefilter('ignore', DeprecationWarning)

                self.setup_distrib_idxs()
            else:
                self.setup_distrib()
            # now update our distrib_size metadata for any distributed
            # unknowns
            sizes = []
            names = []
            for name, meta in iteritems(self._init_unknowns_dict):
                if 'src_indices' in meta:
                    sizes.append(len(meta['src_indices']))
                    names.append(name)
            if sizes:
                if trace:   # pragma: no cover
                    debug("allgathering src index sizes:")
                allsizes = np.zeros((self.comm.size, len(sizes)), dtype=int)
                self.comm.Allgather(np.array(sizes, dtype=int), allsizes)
                for i, name in enumerate(names):
                    self._init_unknowns_dict[name]['distrib_size'] = np.sum(allsizes[:, i])

        # key with absolute path names and add promoted names
        self._params_dict = OrderedDict()
        for name, meta in iteritems(self._init_params_dict):
            pathname = self._get_var_pathname(name)
            self._params_dict [pathname] = meta
            meta['pathname'] = pathname
            to_prom_pname[pathname] = name
            to_abs_pnames[name] = (pathname,)

        self._unknowns_dict = OrderedDict()
        for name, meta in iteritems(self._init_unknowns_dict):
            pathname = self._get_var_pathname(name)
            self._unknowns_dict[pathname] = meta
            meta['pathname'] = pathname
            to_prom_uname[pathname] = name
            to_abs_uname[name] = pathname

        to_prom_name.update(to_prom_uname)
        to_prom_name.update(to_prom_pname)

        self._post_setup_vars = True

        self._sysdata._params_dict = self._params_dict
        self._sysdata._unknowns_dict = self._unknowns_dict

        return self._params_dict, self._unknowns_dict

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign communicator to this `Component`.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.

        parent_dir : str
            The absolute directory of the parent, or '' if unspecified. Used to
            determine the absolute directory of all FileRefs.

        """
        super(Component, self)._setup_communicators(comm, parent_dir)

        # set absolute directories of any FileRefs
        for meta in chain(itervalues(self._init_unknowns_dict),
                          itervalues(self._init_params_dict)):
            val = meta['val']
            #if var is a FileRef, set its absolute directory
            if isinstance(val, FileRef):
                self._fileref_setup(val)

        if not self.is_active():
            for meta in itervalues(self._init_params_dict):
                meta['remote'] = True
            for meta in itervalues(self._init_unknowns_dict):
                meta['remote'] = True

    def _fileref_setup(self, fref):
        fref.parent_dir = self._sysdata.absdir
        d = fref._abspath()
        if self.is_active() and not os.path.exists(os.path.dirname(d)):
            if self.create_dirs:
                os.makedirs(os.path.dirname(d))
            else:
                raise RuntimeError("directory '%s' doesn't exist "
                                   "for FileRef('%s'). Set create_dirs=True "
                                   "in system '%s' to create the directory "
                                   "automatically." %
                                   (os.path.dirname(d),
                                   fref.fname, self.pathname))

    def _setup_vectors(self, param_owners, parent,
                       top_unknowns=None, impl=None, alloc_derivs=True):
        """
        Set up local `VecWrappers` to store this component's variables.

        Args
        ----
        param_owners : dict
            a dictionary mapping `System` pathnames to the pathnames of
            parameters they are reponsible for propagating. (ignored)

        parent : `Group`
            The parent `Group`.

        top_unknowns : `VecWrapper`, optional
            the `Problem` level unknowns `VecWrapper`

        impl : an implementation factory, optional
            Specifies the factory object used to create `VecWrapper` objects.

        alloc_derivs : bool(True)
            If True, allocate the derivative vectors.
        """
        self.params = self.unknowns = self.resids = None
        self.dumat, self.dpmat, self.drmat = OrderedDict(), OrderedDict(), OrderedDict()
        self.connections = self._probdata.connections

        relevance = self._probdata.relevance

        if not self.is_active():
            return

        self._impl = impl

        # create map of relative name in parent to relative name in child
        self._relname_map = self._get_relname_map(parent._sysdata.to_prom_name)

        # at the Group level, we create a set of arrays for each variable of
        # interest, and we make them all subviews of the same shared array in
        # order to conserve memory. Components don't actually own their params,
        # so we just use an empty shared array for dp (with an offset of 0)
        self._shared_dp_vec = empty_arr
        self._shared_p_offsets = { None:0 }
        for vois in chain(relevance.inputs, relevance.outputs):
            for voi in vois:
                self._shared_p_offsets[voi] = 0

        # we don't get non-deriv vecs (u, p, r) unless we have a None group,
        # so force their creation here
        self._create_views(top_unknowns, parent, [], None)

        all_vois = set([None])
        if self._probdata.top_lin_gs: # only need voi vecs for lings
            # create storage for the relevant vecwrappers, keyed by
            # variable_of_interest
            for vois in relevance.groups:
                all_vois.update(vois)
                for voi in vois:
                    self._create_views(top_unknowns, parent, [], voi)

        # create params vec entries for any unconnected params
        for meta in itervalues(self._params_dict):
            pathname = meta['pathname']
            name = self._sysdata._scoped_abs_name(pathname)
            if name not in self.params:
                self.params._add_unconnected_var(pathname, meta)

    def _sys_apply_nonlinear(self, params, unknowns, resids):
        """
        Evaluates the residuals for this component. This wraps
        apply_nonlinear and performs any necessary pre/post operations.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)
        """
        self.apply_nonlinear(params, unknowns, resids)
        resids._scale_values()

    def apply_nonlinear(self, params, unknowns, resids):
        """
        Evaluates the residuals for this component. For explicit
        components, the residual is the output produced by the current params
        minus the previously calculated output. Thus, an explicit component
        must execute its solve nonlinear method. Implicit components should
        override this and calculate their residuals in place.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)
        """

        # Note, we solve a slightly modified version of the unified
        # derivatives equations in OpenMDAO.
        # (dR/du) * (du/dr) = -I
        # The minus side on the right hand side comes from defining the
        # explicit residual to be ynew - yold instead of yold - ynew. The
        # advantage of this is that the derivative of an explicit residual is
        # the same sign as the derivative of the explicit unknown.

        # Since explicit comps don't put anything in resids, we can use it to
        # cache the old values of the unknowns.
        resids.vec[:] = -unknowns.vec

        self._sys_solve_nonlinear(params, unknowns, resids)

        # Unknowns are restored to the old values too. apply_nonlinear does
        # not change the output vector.
        resids.vec[:] += unknowns.vec
        unknowns.vec[:] -= resids.vec

    def _sys_solve_nonlinear(self, params, unknowns, resids):
        """
        Runs the component. This wraps solve_nonlinear and performs any
        necessary pre/post operations.

        Args
        ----
        params : `VecWrapper`, optional
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`, optional
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`, optional
            `VecWrapper` containing residuals. (r)
        """
        unknowns._disable_scaling()
        self.solve_nonlinear(params, unknowns, resids)
        unknowns._scale_values()

    def solve_nonlinear(self, params, unknowns, resids):
        """
        Runs the component. The user is required to define this function in
        all components.

        Args
        ----
        params : `VecWrapper`, optional
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`, optional
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`, optional
            `VecWrapper` containing residuals. (r)
        """

        msg = "Class '%s' does not implement 'solve_nonlinear'"
        raise NotImplementedError(msg  % self.__class__.__name__)

    def linearize(self, params, unknowns, resids):
        """
        Returns Jacobian. Returns None unless component overides this method
        and returns something. J should be a dictionary whose keys are tuples
        of the form ('unknown', 'param') and whose values are ndarrays.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """
        return None

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """
        Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode). If the user doesn't provide this
        method, then we just multiply by the cached jacobian.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        dparams : `VecWrapper`
            `VecWrapper` containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns : `VecWrapper`
            In forward mode, this `VecWrapper` contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        dresids : `VecWrapper`
            `VecWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (dr)

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.
        """
        self._apply_linear_jac(params, unknowns, dparams, dunknowns, dresids,
                               mode)

    def solve_linear(self, dumat, drmat, vois, mode=None):
        """
        Single linear solution applied to whatever input is sitting in
        the rhs vector.

        Args
        ----
        dumat : dict of `VecWrappers`
            In forward mode, each `VecWrapper` contains the incoming vector
            for the states. There is one vector per quantity of interest for
            this problem. In reverse mode, it contains the outgoing vector for
            the states. (du)

        drmat : `dict of VecWrappers`
            `VecWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. There is one vector per
            quantity of interest for this problem. (dr)

        vois : list of strings
            List of all quantities of interest to key into the mats.

        mode : string
            Derivative mode, can be 'fwd' or 'rev', but generally should be
            called without mode so that the user can set the mode in this
            system's ln_solver.options.
        """

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        for voi in vois:
            sol_vec[voi].vec[:] = -rhs_vec[voi].vec

    def dump(self, nest=0, out_stream=sys.stdout, verbose=False, dvecs=False,
             sizes=False):
        """
        Writes a formated dump of this `Component` to file.

        Args
        ----
        nest : int, optional
            Starting nesting level.  Defaults to 0.

        out_stream : an open file, optional
            Where output is written.  Defaults to sys.stdout.

        verbose : bool, optional
            If True, output additional info beyond
            just the tree structure. Default is False.

        dvecs : bool, optional
            If True, show contents of du and dp vectors instead of
            u and p (the default).

        sizes : bool, optional
            If True, show sizes of vectors and comms. Default is False.
        """
        klass = self.__class__.__name__
        if dvecs:
            ulabel, plabel, uvecname, pvecname = 'du', 'dp', 'dunknowns', 'dparams'
        else:
            ulabel, plabel, uvecname, pvecname = 'u', 'p', 'unknowns', 'params'

        uvec = getattr(self, uvecname)
        pvec = getattr(self, pvecname)

        template = "%s %s '%s'"
        out_stream.write(template % (" "*nest, klass, self.name))

        if sizes:
            commsz = self.comm.size if hasattr(self.comm, 'size') else 0
            template = "    req: %s  usize:%d  psize:%d  commsize:%d"
            out_stream.write(template % (self.get_req_procs(),
                                         uvec.vec.size,
                                         pvec.vec.size,
                                         commsz))
        out_stream.write("\n")

        if verbose:  # pragma: no cover
            lens = [len(n) for n in uvec]
            nwid = max(lens) if lens else 12

            for v in uvec:
                if v in uvec._dat and uvec._dat[v].slice is not None:
                    uslice = '{0}[{1[0]}:{1[1]}]'.format(ulabel,
                                                         uvec._dat[v].slice)
                    tem = "{0}{1:<{nwid}} {2:<21} {3:>10}\n"
                    out_stream.write(tem.format(" "*(nest+8), v, uslice,
                                                repr(uvec[v]), nwid=nwid))
                elif not dvecs: # deriv vecs don't have passing by obj
                    tem = "{0}{1:<{nwid}}  (by_obj) ({2})\n"
                    out_stream.write(tem.format(" "*(nest+8), v, repr(uvec[v]),
                                                nwid=nwid))

        out_stream.flush()

    def _get_relname_map(self, parent_proms):
        """
        Args
        ----
        parent_proms : `dict`
            A dict mapping absolute names to promoted names in the parent
            system.

        Returns
        -------
        dict
            Maps promoted name in parent (owner of unknowns) to
            the corresponding promoted name in the child.
        """
        # unknowns_dict is keyed on absolute pathname

        # use an ordered dict here so we can use this smaller dict when looping
        # during get_view.
        #   (the order of this one matches the order in the parent)
        umap = OrderedDict()

        for key, meta in iteritems(self._init_unknowns_dict):
            # promoted and _init_unknowns_dict key are same
            umap[parent_proms['.'.join((self.pathname, key))]] = key

        return umap

    def complex_step_jacobian(self, params, unknowns, resids, total_derivs=False,
                              fd_params=None, fd_states=None, fd_unknowns=None,
                              poi_indices=None, qoi_indices=None):
        """ Return derivatives of all unknowns in this system w.r.t. all
        incoming params using complex step.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        total_derivs : bool, optional
            Should always be False, as componentwise derivatives only need partials.

        fd_params : list of strings, optional
            List of parameter name strings with respect to which derivatives
            are desired. This is used by problem to limit the derivatives that
            are taken.

        fd_unknowns : list of strings, optional
            List of output or state name strings for derivatives to be
            calculated. This is used by problem to limit the derivatives that
            are taken.

        fd_states : list of strings, optional
            List of state name strings for derivatives to be taken with respect to.
            This is used by problem to limit the derivatives that are taken.

        poi_indices: dict of list of integers, optional
            Should be an empty list, as there is no subcomponent relevance reduction.

        qoi_indices: dict of list of integers, optional
            Should be an empty list, as there is no subcomponent relevance reduction.

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays containing the derivative for that
            tuple pair.
        """

        # Params and Unknowns that we provide at this level.
        if fd_params is None:
            fd_params = self._get_fd_params()
        if fd_unknowns is None:
            fd_unknowns = self._get_fd_unknowns()

        # Use settings in the system dict unless variables override.
        step_size = self.fd_options.get('step_size', 1.0e-6)

        jac = {}
        csparams = ComplexStepTgtVecWrapper(params)
        csunknowns = ComplexStepSrcVecWrapper(unknowns)
        csresids = ComplexStepSrcVecWrapper(resids)

        # Pull result from resids only if comp overrides apply_nonlinear
        states = self.states
        if len(states) > 0:
            resultvec = csresids
        else:
            resultvec = csunknowns

        # Manual override of states.
        if fd_states is not None:
            states = fd_states

        # Compute gradient for this param or state.
        for p_name in chain(fd_params, states):


            # States are stepped in unknowns, not params
            if p_name in states:
                stepvec = csunknowns
                target_input = unknowns._dat[p_name].val
            else:
                stepvec = csparams
                target_input = params._dat[p_name].val

            stepvec.set_complex_var(p_name)

            # promoted names and _init_params_dict keys are same
            mydict = self._init_params_dict.get(p_name, {})

            # Local settings for this var trump all
            fdstep = mydict.get('step_size', step_size)

            # Size our Inputs
            p_size = np.size(target_input)
            p_idxs = range(p_size)

            # Size our Outputs
            for u_name in fd_unknowns:
                u_size = np.size(unknowns[u_name])
                jac[u_name, p_name] = np.zeros((u_size, p_size))

            # apply Complex Step on each index in array
            for j, idx in enumerate(p_idxs):

                stepvec.step_complex(idx, fdstep)
                self._sys_apply_nonlinear(csparams, csunknowns, csresids)

                stepvec.step_complex(idx, -fdstep)

                for u_name in fd_unknowns:
                    result = resultvec.flat(u_name)
                    jac[u_name, p_name][:, j] = result.imag/fdstep

            # Need to clear this out because our next input might be a
            # different vector (state vs param)
            stepvec.set_complex_var(None)

        return jac

    def alloc_jacobian(self):
        """
        Creates a jacobian dictionary with the keys pre-populated and correct
        array sizes allocated. caches the result in the component, and
        returns that cache if it finds it.

        Returns
        -----------
        dict
            pre-allocated jacobian dictionary
        """

        if self._jacobian_cache is not None and len(self._jacobian_cache) > 0:
            return self._jacobian_cache

        self._jacobian_cache = jac = {}

        u_vec = self.unknowns
        p_vec = self.params
        states = self.states

        # Caching while caching
        p_size_storage = [(n, m['size']) for n,m in iteritems(p_vec)
                            if not m.get('pass_by_obj') and not m.get('remote')]

        s_size_storage = []
        u_size_storage = []
        for n, meta in iteritems(u_vec):
            if meta.get('pass_by_obj') or meta.get('remote'):
                continue
            if meta.get('state'):
                s_size_storage.append((n, meta['size']))
            u_size_storage.append((n, meta['size']))

        for u_var, u_size in u_size_storage:
            for p_var, p_size in p_size_storage:
                jac[u_var, p_var] = np.zeros((u_size, p_size))

            for s_var, s_size in s_size_storage:
                jac[u_var, s_var] = np.zeros((u_size, s_size))

        return jac

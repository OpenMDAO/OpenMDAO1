""" Base class for all systems in OpenMDAO."""

import sys
import os
from fnmatch import fnmatch
from itertools import chain
import warnings

from six import string_types, iteritems, itervalues, iterkeys

import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.util.options import OptionsDictionary
from collections import OrderedDict
from openmdao.core.vec_wrapper import VecWrapper
from openmdao.core.vec_wrapper import _PlaceholderVecWrapper
from openmdao.util.type_util import real_types
from openmdao.util.string_util import name_relative_to

trace = os.environ.get('OPENMDAO_TRACE')
if trace:  # pragma: no cover
    from openmdao.core.mpi_wrap import debug

class _SysData(object):
    """A container for System level data that is shared with
    VecWrappers in this System.
    """
    def __init__(self, pathname):
        self.pathname = pathname

        # map absolute name to local promoted name
        self.to_prom_name = {}

        self.to_abs_uname = OrderedDict()  # promoted name to abs name
        self.to_prom_uname = OrderedDict() # abs name to promoted name
        self.to_abs_pnames = OrderedDict()  # promoted name to list of abs names
        self.to_prom_pname = OrderedDict() # abs name to promoted namep


class System(object):
    """ Base class for systems in OpenMDAO. When building models, user should
    inherit from `Group` or `Component`
    """

    def __init__(self):
        self.name = ''
        self.pathname = ''

        self._subsystems = OrderedDict()

        self._params_dict = OrderedDict()
        self._unknowns_dict = OrderedDict()

        # specify which variables are promoted up to the parent.  Wildcards
        # are allowed.
        self._promotes = ()

        self.comm = None

        # create placeholders for all of the vectors
        self.unknowns = _PlaceholderVecWrapper('unknowns')
        self.resids = _PlaceholderVecWrapper('resids')
        self.params = _PlaceholderVecWrapper('params')
        self.dunknowns = _PlaceholderVecWrapper('dunknowns')
        self.dresids = _PlaceholderVecWrapper('dresids')

        opt = self.fd_options = OptionsDictionary()
        opt.add_option('force_fd', False,
                       desc="Set to True to finite difference this system.")
        opt.add_option('form', 'forward',
                       values=['forward', 'backward', 'central', 'complex_step'],
                       desc="Finite difference mode. (forward, backward, central) "
                       "You can also set to 'complex_step' to peform the complex "
                       "step method if your components support it.")
        opt.add_option("step_size", 1.0e-6, lower=0.0,
                       desc="Default finite difference stepsize")
        opt.add_option("step_type", 'absolute',
                       values=['absolute', 'relative'],
                       desc='Set to absolute, relative')

        self._impl = None

        self._num_par_fds = 1 # this will be >1 for ParallelFDGroup
        self._par_fd_id = 0 # for ParallelFDGroup, this will be >= 0 and
                            # <= the number of parallel FDs

        self._reset() # initialize some attrs that are set during setup

    def _reset(self):
        """This is called at the beginning of the problem setup."""
        self.pathname = ''

        self._sysdata = _SysData('')

        # dicts of vectors used for parallel solution of multiple RHS
        self.dumat = {}
        self.dpmat = {}
        self.drmat = {}

        self._local_subsystems = []
        self._fd_params = None

    def __getitem__(self, name):
        """
        Return the variable of the given name from this system.

        Args
        ----
        name : str
            The name of the variable.

        Returns
        -------
        value
            The unflattened value of the given variable.
        """
        msg = "Variable '%s' must be accessed from a containing Group"
        raise RuntimeError(msg % name)

    def _promoted(self, name):
        """Determine if the given variable name is being promoted from this
        `System`.

        Args
        ----
        name : str
            The name of a variable, relative to this `System`.

        Returns
        -------
        bool
            True if the named variable is being promoted from this `System`.

        Raises
        ------
        TypeError
            if the promoted variable specifications are not in a valid format
        """
        if isinstance(self._promotes, string_types):
            raise TypeError("'%s' promotes must be specified as a list, "
                            "tuple or other iterator of strings, but '%s' was specified" %
                            (self.name, self._promotes))

        abs_unames = self._sysdata.to_abs_uname
        abs_pnames = self._sysdata.to_abs_pnames

        mybool = False
        for prom in self._promotes:
            if fnmatch(name, prom):
                if name in abs_pnames or name in abs_unames:
                    return True

        return False

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems found
        with the current configuration of this ``System``.

        Args
        ----
        out_stream : a file-like object, optional
            Stream where report will be written.
        """
        pass

    def _check_promotes(self):
        """Check that the `System`s promotes are valid. Raise an Exception if there
        are any promotes that do not match at least one variable in the `System`.

        Raises
        ------
        TypeError
            if the promoted variable specifications are not in a valid format

        RuntimeError
            if a promoted variable specification does not match any variables
        """
        if isinstance(self._promotes, string_types):
            raise TypeError("'%s' promotes must be specified as a list, "
                            "tuple or other iterator of strings, but '%s' was specified" %
                            (self.name, self._promotes))

        to_prom_name = self._sysdata.to_prom_name
        for prom in self._promotes:
            for name in chain(self._params_dict, self._unknowns_dict):
                pname = to_prom_name[name]
                if fnmatch(pname, prom):
                    break
            else:
                msg = "'%s' promotes '%s' but has no variables matching that specification"
                raise RuntimeError(msg % (self.pathname, prom))

    def subsystems(self, local=False, recurse=False, include_self=False):
        """ Returns an iterator over subsystems.  For `System`, this is an empty list.

        Args
        ----
        local : bool, optional
            If True, only return those `Components` that are local. Default is False.

        recurse : bool, optional
            If True, return all `Components` in the system tree, subject to
            the value of the local arg. Default is False.

        typ : type, optional
            If a class is specified here, only those subsystems that are instances
            of that type will be returned.  Default type is `System`.

        include_self : bool, optional
            If True, yield self before iterating over subsystems, assuming type
            of self is appropriate. Default is False.

        Returns
        -------
        iterator
            Iterator over subsystems.
        """
        if include_self:
            yield self

    def _init_sys_data(self, parent_path, probdata):
        """Set the absolute pathname of each `System` in the tree.

        Parameter
        ---------
        parent_path : str
            The pathname of the parent `System`, which is to be prepended to the
            name of this child `System`.

        probdata : `_ProbData`
            Problem level data container.
        """
        self._reset()

        if parent_path:
            self.pathname = '.'.join((parent_path, self.name))
        else:
            self.pathname = self.name

        self._sysdata = _SysData(self.pathname)
        self._probdata = probdata

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
        pass

    def is_active(self):
        """
        Returns
        -------
        bool
            If running under MPI, returns True if this `System` has a valid
            communicator. Always returns True if not running under MPI.
        """
        return MPI is None or not (self.comm is None or
                                   self.comm == MPI.COMM_NULL)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the min and max
            processors usable by this `System`.
        """
        return (1, 1)

    def _setup_communicators(self, comm):
        """
        Assign communicator to this `System` and all of its subsystems.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.
        """
        minp, maxp = self.get_req_procs()
        if MPI and comm is not None and comm != MPI.COMM_NULL and comm.size < minp:
            raise RuntimeError("%s needs %d MPI processes, but was given only %d." %
                              (self.pathname, minp, comm.size))

        self.comm = comm

    def _set_vars_as_remote(self):
        """
        Set 'remote' attribute in metadata of all variables for this subsystem.
        """
        for meta in itervalues(self._params_dict):
            meta['remote'] = True

        for meta in itervalues(self._unknowns_dict):
            meta['remote'] = True

    def fd_jacobian(self, params, unknowns, resids, total_derivs=False,
                    fd_params=None, fd_unknowns=None,
                    poi_indices=None, qoi_indices=None):
        """Finite difference across all unknowns in this system w.r.t. all
        incoming params.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        total_derivs : bool, optional
            Set to true to calculate total derivatives. Otherwise, partial
            derivatives are returned.

        fd_params : list of strings, optional
            List of parameter name strings with respect to which derivatives
            are desired. This is used by problem to limit the derivatives that
            are taken.

        fd_unknowns : list of strings, optional
            List of output or state name strings for derivatives to be
            calculated. This is used by problem to limit the derivatives that
            are taken.

        poi_indices: dict of list of integers, optional
            This is a dict that contains the index values for each parameter of
            interest, so that we only finite difference those indices.

        qoi_indices: dict of list of integers, optional
            This is a dict that contains the index values for each quantity of
            interest, so that the finite difference is returned only for those
            indices.

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

        abs_pnames = self._sysdata.to_abs_pnames

        # Use settings in the system dict unless variables override.
        step_size = self.fd_options.get('step_size', 1.0e-6)
        form = self.fd_options.get('form', 'forward')
        step_type = self.fd_options.get('step_type', 'relative')

        jac = {}
        cache2 = None

        # Prepare for calculating partial derivatives or total derivatives
        if total_derivs:
            run_model = self.solve_nonlinear
            resultvec = unknowns
            states = ()
        else:
            run_model = self.apply_nonlinear
            resultvec = resids
            states = self.states

        cache1 = resultvec.vec.copy()

        gather_jac = False

        fd_count = -1

        # if doing parallel FD, we need to save results during calculation
        # and then pass them around.  fd_cols stores the
        # column data keyed by (uname, pname, col_id).
        fd_cols = {}

        to_prom_name = self._sysdata.to_prom_name

        # Compute gradient for this param or state.
        for p_name in chain(fd_params, states):

            # If our input is connected to a IndepVarComp, then we need to twiddle
            # the unknowns vector instead of the params vector.
            src = self.connections.get(p_name)
            if src is not None:
                param_src = src[0]  # just the name

                # Have to convert to promoted name to key into unknowns
                if param_src not in self.unknowns:
                    param_src = to_prom_name[param_src]

                target_input = unknowns._dat[param_src].val
            else:
                # Cases where the IndepVarComp is somewhere above us.
                if p_name in states:
                    inputs = unknowns
                else:
                    inputs = params

                target_input = inputs._dat[p_name].val

            mydict = {}
            # since p_name is a promoted name, it could refer to multiple
            # params.  We've checked earlier to make sure that step_size,
            # step_type, and form are not defined differently for each
            # matching param.  If they differ, a warning has already been issued.
            if p_name in abs_pnames:
                mydict = self._params_dict[abs_pnames[p_name][0]]

            # Local settings for this var trump all
            fdstep = mydict.get('step_size', step_size)
            fdtype = mydict.get('step_type', step_type)
            fdform = mydict.get('form', form)

            # Size our Inputs
            if poi_indices and param_src in poi_indices:
                p_idxs = poi_indices[param_src]
                p_size = len(p_idxs)
            else:
                p_size = np.size(target_input)
                p_idxs = range(p_size)

            # Size our Outputs
            for u_name in fd_unknowns:
                if qoi_indices and u_name in qoi_indices:
                    u_size = len(qoi_indices[u_name])
                else:
                    u_size = np.size(unknowns[u_name])

                jac[u_name, p_name] = np.zeros((u_size, p_size))

            # if a given param isn't present in this process, we need
            # to still run the model once for each entry in that param
            # in order to stay in sync with the other processes.
            if p_size == 0:
                gather_jac = True
                p_idxs = range(self._params_dict[p_name]['size'])

            # Finite Difference each index in array
            for col, idx in enumerate(p_idxs):
                fd_count += 1

                # skip the current index if its done by some other
                # parallel fd proc
                if fd_count % self._num_par_fds == self._par_fd_id:
                    if p_size == 0:
                        run_model(params, unknowns, resids)
                        continue

                    # Relative or Absolute step size
                    if fdtype == 'relative':
                        step = target_input[idx] * fdstep
                        if step < fdstep:
                            step = fdstep
                    else:
                        step = fdstep

                    if fdform == 'forward':

                        target_input[idx] += step

                        run_model(params, unknowns, resids)

                        target_input[idx] -= step

                        # delta resid is delta unknown
                        resultvec.vec[:] -= cache1
                        resultvec.vec[:] *= (1.0/step)

                    elif fdform == 'backward':

                        target_input[idx] -= step

                        run_model(params, unknowns, resids)

                        target_input[idx] += step

                        # delta resid is delta unknown
                        resultvec.vec[:] -= cache1
                        resultvec.vec[:] *= (-1.0/step)

                    elif fdform == 'central':

                        target_input[idx] += step

                        run_model(params, unknowns, resids)
                        cache2 = resultvec.vec.copy()

                        target_input[idx] -= step
                        resultvec.vec[:] = cache1

                        target_input[idx] -= step

                        run_model(params, unknowns, resids)

                        # central difference formula
                        resultvec.vec[:] -= cache2
                        resultvec.vec[:] *= (-0.5/step)

                        target_input[idx] += step

                    for u_name in fd_unknowns:
                        if qoi_indices and u_name in qoi_indices:
                            result = resultvec._dat[u_name].val[qoi_indices[u_name]]
                        else:
                            result = resultvec._dat[u_name].val
                        jac[u_name, p_name][:, col] = result
                        if self._num_par_fds > 1: # pragma: no cover
                            fd_cols[(u_name, p_name, col)] = \
                                                   jac[u_name, p_name][:, col]

                    # Restore old residual
                    resultvec.vec[:] = cache1

        if self._num_par_fds > 1:
            if trace:  # pragma: no cover
                debug("%s: allgathering parallel FD columns" % self.pathname)
            jacinfos = self._full_comm.allgather(fd_cols)
            for rank, jacinfo in enumerate(jacinfos):
                if rank == self._full_comm.rank:
                    continue
                for key, val in iteritems(jacinfo):
                    if key not in fd_cols:
                        uname, pname, col = key
                        jac[uname, pname][:, col] = val
                        fd_cols[(uname, pname, col)] = val # to avoid setting dups
        elif MPI and gather_jac:
            jac = self.get_combined_jac(jac)

        return jac

    def _sys_apply_linear(self, mode, do_apply, vois=(None,), gs_outputs=None):
        """
        Entry point method for all parent classes to access the apply_linear method.
        This method handles the functionality for self-fd, or otherwise passes the call
        down to the apply_linear method.

        Args
        ----
        mode : string
            Derivative mode, can be 'fwd' or 'rev'.
        vois: list of strings
            List of all quantities of interest to key into the mats.
        do_apply : dict
            We can only solve derivatives for the inputs the instigating
            system has access to.
        gs_outputs : dict, optional
            Linear Gauss-Siedel can limit the outputs when calling apply.
        """
        force_fd = self.fd_options['force_fd']
        states = self.states
        is_relevant = self._probdata.relevance.is_relevant_system
        fwd = mode == "fwd"

        for voi in vois:
            # don't call apply_linear if this system is irrelevant
            if not is_relevant(voi, self):
                continue

            dresids = self.drmat[voi]
            dunknowns = self.dumat[voi]
            dparams = self.dpmat[voi]
            gsouts = None if gs_outputs is None else gs_outputs[voi]

            if fwd:
                dresids.vec[:] = 0.0

                if do_apply[(self.pathname, voi)]:
                    dparams._apply_unit_derivatives()
                    if force_fd:
                        self._apply_linear_jac(self.params, self.unknowns, dparams, dunknowns, dresids, mode)
                    else:
                        self.apply_linear(self.params, self.unknowns, dparams, dunknowns, dresids, mode)
                    dresids.vec *= -1.0

                for var, val in dunknowns.veciter():
                    # Skip all states
                    if (gsouts is None or var in gsouts) and \
                           var not in states:
                        dresids._dat[var].val += val
            else:
                # This zeros out some vars that are not in the local .vec, so we can't just
                # do dparams.vec[:] = 0.0 for example.
                for _, val in dparams.veciter():
                    val[:] = 0.0
                for _, val in dunknowns.veciter():
                    val[:] = 0.0

                if do_apply[(self.pathname, voi)]:
                    try:
                        # Sign on the local Jacobian needs to be -1 before
                        # we add in the fake residual. Since we can't modify
                        # the 'du' vector at this point without stomping on the
                        # previous component's contributions, we can multiply
                        # our local 'arg' by -1, and then revert it afterwards.
                        dresids.vec *= -1.0
                        if force_fd:
                            self._apply_linear_jac(self.params, self.unknowns, dparams, dunknowns, dresids, mode)
                        else:
                            self.apply_linear(self.params, self.unknowns, dparams, dunknowns, dresids, mode)
                        dresids.vec *= -1.0
                    finally:
                        dparams._apply_unit_derivatives()

                for var, val in dresids.veciter():
                    # Skip all states
                    if (gsouts is None or var in gsouts) and \
                            var not in states:
                        dunknowns._dat[var].val += val

    def _sys_linearize(self, params, unknowns, resids, total_derivs=None):
        """
        Entry point for all callers to cause linearization
        of system and all children of system

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        total_derivs: bool
            flag indicating if total or partial derivatives are being forced.
            None allows the system to choose whats appropriate for itself

        """
        if self.fd_options['force_fd']:
            #force_fd should compute semi-totals across all children,
            #    unless total_derivs=False is specifically requested
            if self._local_subsystems and total_derivs is None:
                self._jacobian_cache = self.fd_jacobian(params, unknowns, resids,
                                                        total_derivs=True)
            else:

                # Component can request to use complex step.
                if self.fd_options['form'] == 'complex_step':
                    fd_func = self.complex_step_jacobian
                else:
                    fd_func = self.fd_jacobian
                self._jacobian_cache = fd_func(params, unknowns, resids,
                                               total_derivs=False)

        else:
            try:
                linearize = self.jacobian
            except AttributeError:
                linearize = self.linearize
            else:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn("%s: The 'jacobian' method is deprecated. Please "
                              "rename 'jacobian' to 'linearize'." %
                              self.pathname, DeprecationWarning,stacklevel=2)
                warnings.simplefilter('ignore', DeprecationWarning)

            self._jacobian_cache = linearize(params, unknowns, resids)

        if self._jacobian_cache is not None:
            jc = self._jacobian_cache
            for key, J in iteritems(jc):
                if isinstance(J, real_types):
                    jc[key] = np.array([[J]])
                shape = jc[key].shape
                if len(shape) < 2:
                    jc[key] = jc[key].reshape((shape[0], 1))

        return self._jacobian_cache

    def _apply_linear_jac(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ See apply_linear. This method allows the framework to override
        any derivative specification in any `Component` or `Group` to perform
        finite difference."""

        if not self._jacobian_cache:
            msg = ("No derivatives defined for Component '{name}'")
            msg = msg.format(name=self.pathname)
            raise ValueError(msg)

        isvw = isinstance(dresids, VecWrapper)
        fwd = mode == 'fwd'
        try:
            states = self.states
        except AttributeError:  # handle component unit test where setup has not been performed
            # TODO: should we force all component unit tests to use a Problem test harness?
            states = set([p for u,p in iterkeys(self._jacobian_cache)
                             if p not in dparams])

        for (unknown, param), J in iteritems(self._jacobian_cache):
            if param in states:
                arg_vec = dunknowns
            else:
                arg_vec = dparams

            # Vectors are flipped during adjoint

            try:
                if isvw:
                    if fwd:
                        vec = dresids._flat(unknown)
                        vec += J.dot(arg_vec._flat(param))
                    else:
                        shape = arg_vec._dat[param].meta['shape']
                        arg_vec[param] += J.T.dot(dresids._flat(unknown)).reshape(shape)
                else: # plain dicts were passed in for unit testing...
                    if fwd:
                        vec = dresids[unknown]
                        vec += J.dot(arg_vec[param].flat).reshape(vec.shape)
                    else:
                        shape = arg_vec[param].shape
                        arg_vec[param] += J.T.dot(dresids[unknown].flat).reshape(shape)
            except KeyError:
                continue # either didn't find param in dparams/dunknowns or
                         # didn't find unknown in dresids
            except ValueError:
                # Provide a user-readable message that locates the problem
                # derivative term.
                req_shape = (len(dresids[unknown].flat), len(arg_vec[param]))
                msg = "In component '{}', the derivative of '{}' wrt '{}' should have shape '{}' "
                msg += "but has shape '{}' instead."
                msg = msg.format(self.pathname, unknown, param, req_shape, J.shape)
                raise ValueError(msg)

    def _create_views(self, top_unknowns, parent, my_params,
                      voi=None):
        """
        A manager of the data transfer of a possibly distributed collection of
        variables.  The variables are based on views into an existing
        `VecWrapper`.

        Args
        ----
        top_unknowns : `VecWrapper`
            The `Problem` level unknowns `VecWrapper`.

        parent : `System`
            The `System` which provides the `VecWrapper` on which to create views.

        my_params : list
            List of pathnames for parameters that this `Group` is
            responsible for propagating.

        relevance : `Relevance`
            Object containing relevance info for each variable of interest.

        voi : str
            The name of a variable of interest.

        """

        comm = self.comm
        params_dict = self._params_dict
        relevance = self._probdata.relevance

        # map promoted name in parent to corresponding promoted name in this view
        umap = self._relname_map

        if voi is None:
            self.unknowns = parent.unknowns.get_view(self, comm, umap)
            self.states = set((n for n,m in iteritems(self.unknowns) if m.get('state')))
            self.resids = parent.resids.get_view(self, comm, umap)
            self.params = parent._impl.create_tgt_vecwrapper(self._sysdata, comm)
            self.params.setup(parent.params, params_dict, top_unknowns,
                              my_params, self.connections, relevance=relevance,
                              store_byobjs=True)

        self.dumat[voi] = parent.dumat[voi].get_view(self, comm, umap)
        self.drmat[voi] = parent.drmat[voi].get_view(self, comm, umap)
        self.dpmat[voi] = parent._impl.create_tgt_vecwrapper(self._sysdata, comm)

        self.dpmat[voi].setup(parent.dpmat[voi], params_dict, top_unknowns,
                  my_params, self.connections,
                  relevance=relevance, var_of_interest=voi,
                  shared_vec=self._shared_dp_vec[self._shared_p_offsets[voi]:])

    def get_combined_jac(self, J):
        """
        Take a J dict that's distributed, i.e., has different values across
        different MPI processes, and return a dict that contains all of the
        values from all of the processes. If values are duplicated, use the
        value from the lowest rank process. Note that J has a nested dict
        structure.

        Args
        ----
        J : `dict`
            Local Jacobian

        Returns
        -------
        `dict`
            Local gathered Jacobian
        """

        if not self.is_active():
            return J

        comm = self.comm
        iproc = comm.rank

        # TODO: calculate dist_need_tups and dist_has_tups once
        #       and cache it instead of doing every time.
        need_tups = []
        has_tups = []

        # Gather a list of local tuples for J.
        for (output, param), value in iteritems(J):
            if value.size == 0:
                need_tups.append((output, param))
            else:
                has_tups.append((output, param))

        if trace:  # pragma: no cover
            debug("%s: allgather of needed tups" % self.pathname)
        dist_need_tups = comm.allgather(need_tups)

        needed_set = set()
        for need_tups in dist_need_tups:
            needed_set.update(need_tups)

        if not needed_set:
            return J  # nobody needs any J entries

        if trace:  # pragma: no cover
            debug("%s: allgather of has_tups" % self.pathname)
        dist_has_tups = comm.allgather(has_tups)

        found = set()
        owned_vals = []
        for rank, tups in enumerate(dist_has_tups):
            for tup in tups:
                if tup in needed_set and not tup in found:
                    found.add(tup)
                    if rank == iproc:
                        owned_vals.append((tup, J[tup]))

        if trace:  # pragma: no cover
            debug("%s: allgather of owned vals" % self.pathname)
        dist_vals = comm.allgather(owned_vals)

        for rank, vals in enumerate(dist_vals):
            if rank != iproc:
                for (output, param), value in vals:
                    J[output, param] = value

        return J

    def _get_var_pathname(self, name):
        if self.pathname:
            return '.'.join((self.pathname, name))
        return name

    def generate_docstring(self):
        """
        Generates a numpy-style docstring for a user-created System class.

        Returns
        -------
        docstring : str
                string that contains a basic numpy docstring.

        """
        #start the docstring off
        docstring = '    \"\"\"\n'

        if self._init_params_dict or self._init_unknowns_dict:
            docstring += '\n    Params\n    ----------\n'

        if self._init_params_dict:
            for key, value in self._init_params_dict.items():
                #docstring += type(value).__name__
                docstring += "    " + key + ": param ({"
                #get the values out in order
                dictItemCount = len(value)
                dictPosition = 1
                for k in sorted(value):
                    docstring +=  "'" +k+ "'" + ": " + str(value[k])
                    #don't want a trailing comma
                    if (dictPosition != dictItemCount):
                        docstring += ", "
                    dictPosition += 1
                docstring += "})\n"

        if self._init_unknowns_dict:
            for key, value in self._init_unknowns_dict.items():
                docstring += "    " + key + " : unknown ({"
                dictItemCount = len(value)
                dictPosition = 1
                for k in sorted(value):
                    docstring += "'" +k+ "'" + ": " + str(value[k])
                    if (dictPosition != dictItemCount):
                        docstring += ", "
                    dictPosition += 1
                docstring += "})\n"

        #Put options into docstring
        firstTime = 1
        #for py3.4, items from vars must come out in same order.
        v = OrderedDict(sorted(vars(self).items()))
        for key, value in v.items():
            if type(value)==OptionsDictionary:
                if firstTime:  #start of Options docstring
                    docstring += '\n    Options\n    -------\n'
                    firstTime = 0
                for (name, val) in sorted(value.items()):
                    docstring += "    " + key + "['"
                    docstring += name + "']"
                    docstring += " :  " + type(val).__name__
                    docstring += "("
                    if type(val).__name__ == 'str':
                        docstring += "'"
                    docstring += str(val)
                    if type(val).__name__ == 'str':
                        docstring += "'"
                    docstring += ")\n"

                    desc = value._options[name]['desc']
                    if(desc):
                        docstring += "        " + desc + "\n"

        #finish up docstring
        docstring += '\n    \"\"\"\n'
        return docstring

    def _get_shared_vec_info(self, vdict, my_params=None):
        # determine the size of the largest grouping of parallel subvecs and the
        # offsets within those vecs for each voi in a parallel set.
        # We should never need more memory than the largest sized collection of
        # parallel vecs.
        if my_params is None:
            metas = [m for m in itervalues(vdict)
                          if not m.get('pass_by_obj')]
        else: # for params, we only include 'owned' vars in the vector
            metas = [m for m in itervalues(vdict)
                          if m['pathname'] in my_params and
                             not m.get('pass_by_obj')]

        full_size = sum([m['size'] for m in metas])  # 'None' vecs are this size
        max_size = full_size

        offsets = { None: 0 }

        # no parallel rhs vecs, so biggest one will just be the one containing all
        # vars.
        if not self._probdata.top_lin_gs:
            return max_size, offsets

        relevance = self._probdata.relevance
        for vois in relevance.groups:
            vec_size = 0
            for voi in vois:
                sz = sum([m['size'] for m in metas
                                 if m['pathname'] in vdict and
                                    relevance.is_relevant(voi, m['top_promoted_name'])])
                offsets[voi] = vec_size
                vec_size += sz

            if vec_size > max_size:
                max_size = vec_size

        return max_size, offsets

    def _setup_prom_map(self):
        """
        Sets up the internal dict that maps absolute name to promoted name.
        """
        to_prom_name = self._sysdata.to_prom_name

        for pathname, meta in iteritems(self._unknowns_dict):
            prom = to_prom_name[pathname]

        for pathname, meta in iteritems(self._params_dict):
            prom = to_prom_name[pathname]

    def list_connections(self, group_by_comp=False, stream=sys.stdout):
        """
        Writes out the list of all connections involving this System or any
        of its children.  The list is of the form:

        source_absolute_name (source_promoted_name) -> target

        Where sources that broadcast to multiple targets will be replaced with
        a blank source for all but the first of their targets, in order to help
        broadcast sources visually stand out.  The source name will be followed
        by its promoted name if it differs, and if a target is promoted it will
        be followed by a '*'.

        Sources are sorted alphabetically and targets are subsorted
        alphabetically when a source is broadcast to multiple targets.

        Args
        ----
        group_by_comp : bool, optional
            If True, show all sources and targets grouped by component. Note
            that this will cause repeated lines in the output since a given
            connection will always be from one component's source to a different
            component's target.  Default is False.

        stream : output stream, optional
            Stream to write the connection info to. Defaults to sys.stdout.
        """

        def _list_conns(self):
            template = "{0:<{swid}} -> {1}\n"

            to_prom_name = self._probdata.to_prom_name
            scope = self.pathname + '.' if self.pathname else ''

            # create a dict with srcs as keys so we can more easily subsort
            # targets after sorting srcs.
            by_src = {}
            for tgt, (src, idx) in iteritems(self.connections):
                if src.startswith(scope) or tgt.startswith(scope):
                    if to_prom_name[tgt] != tgt:
                        tgt += ' *'
                    if to_prom_name[src] != src:
                        src += " (%s)" % to_prom_name[src]
                    by_src.setdefault(src, []).append(tgt)

            src_max_wid = max(len(n) for n in by_src)

            for src, tgts in sorted(iteritems(by_src), key=lambda x: x[0]):
                for i, tgt in enumerate(sorted(tgts)):
                    if i: src = ''
                    stream.write(template.format(src, tgt, swid=src_max_wid))

        if group_by_comp:
            for c in self.components(recurse=True, include_self=True):
                line = "Connections for %s:" % c.pathname
                stream.write("\n%s\n%s\n" % (line, '-'*len(line)))
                c.list_connections(stream=stream)
        else:
            _list_conns(self)

def _iter_J_nested(J):
    for output, subdict in iteritems(J):
        for param, value in iteritems(subdict):
            yield (output, param), value

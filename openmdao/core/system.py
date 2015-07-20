""" Base class for all systems in OpenMDAO."""

import sys
from collections import OrderedDict
from fnmatch import fnmatch
from itertools import chain
from six import string_types, iteritems

import numpy as np

from openmdao.core.mpiwrap import MPI
from openmdao.core.options import OptionsDictionary
from openmdao.core.vecwrapper import PlaceholderVecWrapper


class System(object):
    """ Base class for systems in OpenMDAO. When building models, user should
    inherit from `Group` or `Component`"""

    def __init__(self):
        self.name = ''
        self.pathname = ''

        self._params_dict = OrderedDict()
        self._unknowns_dict = OrderedDict()

        # specify which variables are promoted up to the parent.  Wildcards
        # are allowed.
        self._promotes = ()

        self.comm = None

        # create placeholders for all of the vectors
        self.unknowns = PlaceholderVecWrapper('unknowns')
        self.resids = PlaceholderVecWrapper('resids')
        self.params = PlaceholderVecWrapper('params')
        self.dunknowns = PlaceholderVecWrapper('dunknowns')
        self.dresids = PlaceholderVecWrapper('dresids')
        self.dparams = PlaceholderVecWrapper('dparams')

        # dicts of vectors used for parallel solution of multiple RHS
        self.dumat = {}
        self.dpmat = {}
        self.drmat = {}

        opt = self.fd_options = OptionsDictionary()
        opt.add_option('force_fd', False,
                       desc="Set to True to finite difference this system.")
        opt.add_option('form', 'forward',
                       values=['forward', 'backward', 'central', 'complex_step'],
                       desc="Finite difference mode. (forward, backward, central) "
                       "You can also set to 'complex_step' to peform the complex "
                       "step method if your components support it.")
        opt.add_option("step_size", 1.0e-6,
                       desc="Default finite difference stepsize")
        opt.add_option("step_type", 'absolute',
                       values=['absolute', 'relative'],
                       desc='Set to absolute, relative')

        self._relevance = None
        self._impl_factory = None

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

        for prom in self._promotes:
            if fnmatch(name, prom):
                for meta in chain(self._params_dict.values(),
                                  self._unknowns_dict.values()):
                    if name == meta.get('promoted_name'):
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

        for prom in self._promotes:
            found = False
            for name, meta in chain(self._params_dict.items(), self._unknowns_dict.items()):
                if fnmatch(meta.get('promoted_name', name), prom):
                    found = True
            if not found:
                msg = "'%s' promotes '%s' but has no variables matching that specification"
                raise RuntimeError(msg % (self.name, prom))


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

    def _setup_paths(self, parent_path):
        """Set the absolute pathname of each `System` in the tree.

        Parameter
        ---------
        parent_path : str
            The pathname of the parent `System`, which is to be prepended to the
            name of this child `System`.
        """
        if parent_path:
            self.pathname = '.'.join((parent_path, self.name))
        else:
            self.pathname = self.name

    def clear_dparams(self):
        """ Zeros out the dparams (dp) vector."""

        for parallel_set in self._relevance.vars_of_interest():
            for name in parallel_set:
                if name in self.dpmat:
                    self.dpmat[name].vec[:] = 0.0

        self.dpmat[None].vec[:] = 0.0

        # Recurse to clear all dparams vectors.
        for system in self.subsystems(local=True):
            system.clear_dparams()

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
        return MPI is None or self.comm != MPI.COMM_NULL

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
        self.comm = comm

    def _set_vars_as_remote(self):
        """
        Set 'remote' attribute in metadata of all variables for this subsystem.
        """
        for meta in self._params_dict.values():
            meta['remote'] = True

        for meta in self._unknowns_dict.values():
            meta['remote'] = True

    def fd_jacobian(self, params, unknowns, resids, step_size=None, form=None,
                    step_type=None, total_derivs=False):
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

        step_size : float, optional
            Override all other specifications of finite difference step size.

        form : float, optional
            Override all other specifications of form. Can be forward,
            backward, or central.

        step_type : float, optional
            Override all other specifications of step_type. Can be absolute
            or relative.

        total_derivs : bool, optional
            Set to true to calculate total derivatives. Otherwise, partial
            derivatives are returned.

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays containing the derivative for that
            tuple pair.
        """

        # Params and Unknowns that we provide at this level.
        fd_params = self._get_fd_params()
        fd_unknowns = self._get_fd_unknowns()

        # Function call arguments have precedence over the system dict.
        step_size = self.fd_options.get('step_size', step_size)
        form = self.fd_options.get('form', form)
        step_type = self.fd_options.get('step_type', step_type)

        jac = {}
        cache2 = None

        # Prepare for calculating partial derivatives or total derivatives
        if total_derivs == False:
            run_model = self.apply_nonlinear
            cache1 = resids.vec.copy()
            resultvec = resids
            states = [name for name, meta in self.unknowns.items() if meta.get('state')]
        else:
            run_model = self.solve_nonlinear
            cache1 = unknowns.vec.copy()
            resultvec = unknowns
            states = []

        # Compute gradient for this param or state.
        for p_name in chain(fd_params, states):

            if p_name in states:
                inputs = unknowns
            else:
                inputs = params

            target_input = inputs.flat[p_name]

            # If our input is connected to a Paramcomp, then we need to twiddle
            # the unknowns vector instead of the params vector.
            param_src = self.connections.get(p_name)
            if param_src is not None:

                # Have to convert to promoted name to key into unknowns
                if param_src not in self.unknowns:
                    param_src = self.unknowns.get_promoted_varname(param_src)

                target_input = unknowns.flat[param_src]

            mydict = {}
            for val in self._params_dict.values():
                if val['promoted_name'] == p_name:
                    mydict = val
                    break

            # Local settings for this var trump all
            fdstep = mydict.get('fd_step_size', step_size)
            fdtype = mydict.get('fd_step_type', step_type)
            fdform = mydict.get('fd_form', form)

            # Size our Inputs
            p_size = np.size(inputs[p_name])

            # Size our Outputs
            for u_name in fd_unknowns:
                u_size = np.size(unknowns[u_name])
                jac[u_name, p_name] = np.ones((u_size, p_size))

            # Finite Difference each index in array
            for idx in range(p_size):

                # Relative or Absolute step size
                if fdtype == 'relative':
                    step = inputs.flat[p_name][idx] * fdstep
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
                    jac[u_name, p_name][:, idx] = resultvec.flat[u_name]

                # Restore old residual
                resultvec.vec[:] = cache1

        return jac

    def _apply_linear_jac(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ See apply_linear. This method allows the framework to override
        any derivative specification in any `Component` or `Group` to perform
        finite difference."""

        if not self._jacobian_cache:
            msg = ("No derivatives defined for Component '{name}'")
            msg = msg.format(name=self.name)
            raise ValueError(msg)

        for key, J in iteritems(self._jacobian_cache):
            unknown, param = key

            # States are never in dparams.
            if param in dparams:
                arg_vec = dparams
            elif param in dunknowns:
                arg_vec = dunknowns
            else:
                continue

            if unknown not in dresids:
                continue

            result = dresids[unknown]

            # Vectors are flipped during adjoint

            if mode == 'fwd':
                dresids[unknown] += J.dot(arg_vec[param].flat).reshape(result.shape)
            else:
                arg_vec[param] += J.T.dot(result.flat).reshape(arg_vec[param].shape)

    def _create_vecs(self, my_params, relevance, var_of_interest, impl):
        """ This creates our vecs and mats."""
        comm = self.comm
        sys_pathname = self.pathname
        params_dict = self._params_dict
        unknowns_dict = self._unknowns_dict

        self.comm = comm

        # create implementation specific VecWrappers
        if var_of_interest is None:
            self.unknowns = impl.create_src_vecwrapper(sys_pathname, comm)
            self.resids = impl.create_src_vecwrapper(sys_pathname, comm)
            self.params = impl.create_tgt_vecwrapper(sys_pathname, comm)

            # populate the VecWrappers with data
            self.unknowns.setup(unknowns_dict, store_byobjs=True)
            self.resids.setup(unknowns_dict)
            self.params.setup(None, params_dict, self.unknowns,
                              my_params, self.connections, store_byobjs=True)

        dunknowns = impl.create_src_vecwrapper(sys_pathname, comm)
        dresids = impl.create_src_vecwrapper(sys_pathname, comm)
        dparams = impl.create_tgt_vecwrapper(sys_pathname, comm)

        dunknowns.setup(unknowns_dict, relevant_vars=relevance[var_of_interest])
        dresids.setup(unknowns_dict, relevant_vars=relevance[var_of_interest])
        dparams.setup(None, params_dict, self.unknowns, my_params,
                      self.connections,
                      relevant_vars=relevance[var_of_interest])

        self.dumat[var_of_interest] = dunknowns
        self.drmat[var_of_interest] = dresids
        self.dpmat[var_of_interest] = dparams

    def _create_views(self, top_unknowns, parent, my_params, relevance,
                      var_of_interest=None):
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

        var_of_interest : str
            The name of a variable of interest.

        Returns
        -------
        `VecTuple`
            A namedtuple of six (6) `VecWrappers`:
            unknowns, dunknowns, resids, dresids, params, dparams.
        """

        comm = self.comm
        unknowns_dict = self._unknowns_dict
        params_dict = self._params_dict
        voi = var_of_interest

        # map promoted name in parent to corresponding promoted name in this view
        umap = _get_relname_map(parent.unknowns, unknowns_dict, self.pathname)

        if voi is None:
            self.unknowns = parent.unknowns.get_view(self.pathname, comm, umap, relevance,
                                                     voi)
            self.resids = parent.resids.get_view(self.pathname, comm, umap, relevance,
                                                 voi)
            self.params = parent._impl_factory.create_tgt_vecwrapper(self.pathname, comm)
            self.params.setup(parent.params, params_dict, top_unknowns,
                              my_params, self.connections, store_byobjs=True)

        self.dumat[voi] = parent.dumat[voi].get_view(self.pathname, comm, umap,
                                                     relevance, voi)
        self.drmat[voi] = parent.drmat[voi].get_view(self.pathname, comm, umap,
                                                     relevance, voi)
        self.dpmat[voi] = parent._impl_factory.create_tgt_vecwrapper(self.pathname, comm)
        self.dpmat[voi].setup(parent.dpmat[voi], params_dict, top_unknowns,
                              my_params, self.connections,
                              relevant_vars=relevance[voi])

    #def get_combined_jac(self, J):
        #"""
        #Take a J dict that's distributed, i.e., has different values across
        #different MPI processes, and return a dict that contains all of the
        #values from all of the processes. If values are duplicated, use the
        #value from the lowest rank process. Note that J has a nested dict
        #structure.

        #Args
        #----
        #J : `dict`
            #Distributed Jacobian

        #Returns
        #-------
        #`dict`
            #Local gathered Jacobian
        #"""

        #comm = self.comm
        #if not self.is_active():
            #return J

        #myrank = comm.rank

        #tups = []

        ## Gather a list of local tuples for J.
        #for output, dct in J.items():
            #for param, value in dct.items():

                ## Params are already only on this process. We need to add
                ## only outputs of components that are on this process.
                #sub = getattr(self, output.partition('.')[0])
                #if sub.is_active() and value is not None and value.size > 0:
                    #tups.append((output, param))

        #dist_tups = comm.gather(tups, root=0)

        #tupdict = {}
        #if myrank == 0:
            #for rank, tups in enumerate(dist_tups):
                #for tup in tups:
                    #if not tup in tupdict:
                        #tupdict[tup] = rank

            ##get rid of tups from the root proc before bcast
            #for tup, rank in tupdict.items():
                #if rank == 0:
                    #del tupdict[tup]

        #tupdict = comm.bcast(tupdict, root=0)

        #if myrank == 0:
            #for (param, output), rank in tupdict.items():
                #J[param][output] = comm.recv(source=rank, tag=0)
        #else:
            #for (param, output), rank in tupdict.items():
                #if rank == myrank:
                    #comm.send(J[param][output], dest=0, tag=0)

        ## FIXME: rework some of this using knowledge of local_var_sizes in order
        ## to avoid any unnecessary data passing

        ## return the combined dict
        #return comm.bcast(J, root=0)

    def _get_var_pathname(self, name):
        if self.pathname:
            return '.'.join((self.pathname, name))
        return name

def _get_relname_map(unknowns, unknowns_dict, child_name):
    """
    Args
    ----
    unknowns : `VecWrapper`
        A dict-like object containing variables keyed using promoted names.

    unknowns_dict : `OrderedDict`
        An ordered mapping of absolute variable name to its metadata.

    child_name : str
        The pathname of the child for which to get promoted name.

    Returns
    -------
    dict
        Maps promoted name in parent (owner of unknowns and unknowns_dict) to
        the corresponding promoted name in the child.
    """
    # unknowns is keyed on promoted name relative to the parent system
    # unknowns_dict is keyed on absolute pathname
    umap = {}
    for rel, meta in unknowns.items():
        abspath = meta['pathname']
        if abspath.startswith(child_name+'.'):
            for m in unknowns_dict.values():
                if abspath == m['pathname']:
                    newrel = m['promoted_name']
                    break
            else:
                newrel = rel
            umap[rel] = newrel

    return umap

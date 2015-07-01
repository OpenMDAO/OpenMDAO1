""" Base class for all systems in OpenMDAO."""

from collections import OrderedDict
from fnmatch import fnmatch
from itertools import chain
from six import string_types, iteritems

import numpy as np

from openmdao.core.vecwrapper import PlaceholderVecWrapper
from openmdao.core.mpiwrap import MPI
from openmdao.core.options import OptionsDictionary


class System(object):
    """ Base class for systems in OpenMDAO."""

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
        self.unknowns  = PlaceholderVecWrapper('unknowns')
        self.resids    = PlaceholderVecWrapper('resids')
        self.params    = PlaceholderVecWrapper('params')
        self.dunknowns = PlaceholderVecWrapper('dunknowns')
        self.dresids   = PlaceholderVecWrapper('dresids')
        self.dparams   = PlaceholderVecWrapper('dparams')

        # dicts of vectors used for parallel solution of multiple RHS
        self.dumat = {}
        self.dpmat = {}
        self.drmat = {}

        self.fd_options = OptionsDictionary()
        self.fd_options.add_option('force_fd', False,
                                   desc="Set to True to finite difference this system.")
        self.fd_options.add_option('form', 'forward',
                                   values=['forward', 'backward', 'central', 'complex_step'],
                                   desc="Finite difference mode. (forward, backward, central) "
                                   "You can also set to 'complex_step' to peform the complex "
                                   "step method if your components support it.")
        self.fd_options.add_option("step_size", 1.0e-6,
                                    desc = "Default finite difference stepsize")
        self.fd_options.add_option("step_type", 'absolute',
                                   values=['absolute', 'relative'],
                                   desc='Set to absolute, relative')

    def __getitem__(self, name):
        """
        Return the variable or subsystem of the given name from this system.

        Args
        ----
        name : str
            The name of the variable or subsystem.

        Returns
        -------
        value OR `System`
            The unflattened value of the given variable OR a reference to
            the named `System`.
        """
        raise RuntimeError("Variable '%s' must be accessed from a containing Group" % name)


    def promoted(self, name):
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
        """
        if isinstance(self._promotes, string_types):
            raise TypeError("'%s' promotes must be specified as a list, "
                            "tuple or other iterator of strings, but '%s' was specified" %
                            (self.name, self._promotes))

        for prom in self._promotes:
            if fnmatch(name, prom):
                for n, meta in chain(self._params_dict.items(), self._unknowns_dict.items()):
                    rel = meta.get('promoted_name', n)
                    if rel == name:
                        return True

        return False

    def subsystems(self, local=False, recurse=False):
        """ Returns an iterator over subsystems.  For `System`, this is an empty list.
        """
        return []

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
        for name, system in self.subsystems(local=True):
            system.clear_dparams()

    def preconditioner(self):
        pass

    def jacobian(self, params, unknowns, resids):
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        raise NotImplementedError("solve_nonlinear")

    def apply_nonlinear(self, params, unknowns, resids):
        pass

    def solve_linear(self, params, unknowns, vois, mode="fwd"):
        pass

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode="fwd"):
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
        pname = self.pathname + '.'
        for name, meta in self._params_dict.items():
            if name.startswith(pname):
                meta['remote'] = True

        for name, meta in self._unknowns_dict.items():
            if name.startswith(pname):
                meta['remote'] = True

    def fd_jacobian(self, params, unknowns, resids, step_size=None, form=None,
                    step_type=None, total_derivs=False):
        """Finite difference across all unknowns in this system w.r.t. all
        params.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)

        step_size : float (optional)
            Override all other specifications of finite difference step size.

        form : float (optional)
            Override all other specifications of form. Can be forward,
            backward, or central.

        step_type : float (optional)
            Override all other specifications of step_type. Can be absolute
            or relative.

        total_derivs : bool
            Set to true to calculate total derivatives. Otherwise, partial
            derivatives are returned.

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """

        # Params and Unknowns that we provide at this level.
        fd_params = self._get_fd_params()
        fd_unknowns = self._get_fd_unknowns()

        # Function call arguments have precedence over the system dict.
        if step_size == None:
            step_size = self.fd_options['step_size']
        if form == None:
            form = self.fd_options['form']
        if step_type == None:
            step_type = self.fd_options['step_type']

        jac = {}
        cache2 = None

        # Prepare for calculating partial derivatives or total derivatives
        states = []
        if total_derivs == False:
            run_model = self.apply_nonlinear
            cache1 = resids.vec.copy()
            resultvec = resids
            for u_name, meta in iteritems(self._unknowns_dict):
                if meta.get('state'):
                    states.append(meta['promoted_name'])
        else:
            run_model = self.solve_nonlinear
            cache1 = unknowns.vec.copy()
            resultvec = unknowns

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

                # Have to convert to relative name to key into unknowns
                if param_src not in self.unknowns:
                    for name in unknowns:
                        meta = unknowns.metadata(name)
                        if meta['pathname'] == param_src:
                            param_src = meta['promoted_name']

                target_input = unknowns.flat[param_src]

            mydict = {}
            for key, val in self._params_dict.items():
                if val['promoted_name'] == p_name:
                    mydict = val
                    break

            # Local settings for this var trump all
            if 'fd_step_size' in mydict:
                fdstep = mydict['fd_step_size']
            else:
                fdstep = step_size
            if 'fd_step_type' in mydict:
                fdtype = mydict['fd_step_type']
            else:
                fdtype = step_type
            if 'fd_form' in mydict:
                fdform = mydict['fd_form']
            else:
                fdform = form

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

        if self._jacobian_cache is None:
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
        comm = self.comm
        sys_pathname = self.pathname
        params_dict = self._params_dict
        unknowns_dict = self._unknowns_dict

        self.comm = comm

        # create implementation specific VecWrappers
        if var_of_interest is None:
            self.unknowns  = impl.create_src_vecwrapper(sys_pathname, comm)
            self.resids    = impl.create_src_vecwrapper(sys_pathname, comm)
            self.params    = impl.create_tgt_vecwrapper(sys_pathname, comm)

            # populate the VecWrappers with data
            self.unknowns.setup(unknowns_dict, store_byobjs=True)
            self.resids.setup(unknowns_dict)
            self.params.setup(None, params_dict, self.unknowns,
                              my_params, self.connections, store_byobjs=True)

        dunknowns = impl.create_src_vecwrapper(sys_pathname, comm)
        dresids   = impl.create_src_vecwrapper(sys_pathname, comm)
        dparams   = impl.create_tgt_vecwrapper(sys_pathname, comm)

        dunknowns.setup(unknowns_dict, relevant_vars=relevance[var_of_interest])
        dresids.setup(unknowns_dict, relevant_vars=relevance[var_of_interest])
        dparams.setup(None, params_dict, self.unknowns, my_params, self.connections,
                      relevant_vars=relevance[var_of_interest])

        self.dumat[var_of_interest] = dunknowns
        self.drmat[var_of_interest] = dresids
        self.dpmat[var_of_interest] = dparams

    def _create_views(self, top_unknowns, parent, my_params, relevance, var_of_interest=None):
        """
        A manager of the data transfer of a possibly distributed collection of
        variables.  The variables are based on views into an existing VarManager.

        Args
        ----
        top_unknowns : `VecWrapper`
            The `Problem` level unknowns `VecWrapper`.

        parent : `System`
            The `System` which provides the `VecWrapper` on which to create views.

        my_params : list
            List of pathnames for parameters that this `VarManager` is
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

        # map relative name in parent to corresponding relative name in this view
        umap = get_relname_map(parent.unknowns, unknowns_dict, self.pathname)

        if var_of_interest is None:
            self.unknowns  = parent.unknowns.get_view(self.pathname, comm, umap, relevance,
                                                      var_of_interest)
            self.resids    = parent.resids.get_view(self.pathname, comm, umap, relevance,
                                                    var_of_interest)
            self.params    = parent._impl_factory.create_tgt_vecwrapper(self.pathname, comm)
            self.params.setup(parent.params, params_dict, top_unknowns,
                              my_params, self.connections, store_byobjs=True)

        self.dumat[var_of_interest] = parent.dumat[var_of_interest].get_view(self.pathname, comm, umap,
                                                                             relevance, var_of_interest)
        self.drmat[var_of_interest] = parent.drmat[var_of_interest].get_view(self.pathname, comm, umap,
                                                                             relevance, var_of_interest)
        self.dpmat[var_of_interest] = parent._impl_factory.create_tgt_vecwrapper(self.pathname, comm)
        self.dpmat[var_of_interest].setup(parent.dpmat[var_of_interest], params_dict, top_unknowns,
                                          my_params, self.connections,
                                          relevant_vars=relevance[var_of_interest])

    def get_combined_J(self, J):
        """
        Take a J dict that's distributed, i.e., has different values
        across different MPI processes, and return a dict that
        contains all of the values from all of the processes.  If
        values are duplicated, use the value from the lowest rank
        process.  Note that J has a nested dict structure.
        """

        comm = self.comm
        if not self.is_active():
            return J

        myrank = comm.rank

        tups = []

        # Gather a list of local tuples for J.
        for output, dct in J.items():
            for param, value in dct.items():

                # Params are already only on this process. We need to add
                # only outputs of components that are on this process.
                sys = getattr(self, output.partition('.')[0])
                if sys.is_active() and value is not None and value.size > 0:
                    tups.append((output, param))

        dist_tups = comm.gather(tups, root=0)

        tupdict = {}
        if myrank == 0:
            for rank, tups in enumerate(dist_tups):
                for tup in tups:
                    if not tup in tupdict:
                        tupdict[tup] = rank

            #get rid of tups from the root proc before bcast
            for tup, rank in tupdict.items():
                if rank == 0:
                    del tupdict[tup]

        tupdict = comm.bcast(tupdict, root=0)

        if myrank == 0:
            for (param, output), rank in tupdict.items():
                J[param][output] = comm.recv(source=rank, tag=0)
        else:
            for (param, output), rank in tupdict.items():
                if rank == myrank:
                    comm.send(J[param][output], dest=0, tag=0)

        # FIXME: rework some of this using knowledge of local_var_sizes in order
        # to avoid any unnecessary data passing

        # return the combined dict
        return comm.bcast(J, root=0)

def get_relname_map(unknowns, unknowns_dict, child_name):
    """
    Args
    ----
    unknowns : `VecWrapper`
        A dict-like object containing variables keyed using relative names.

    unknowns_dict : `OrderedDict`
        An ordered mapping of absolute variable name to its metadata.

    child_name : str
        The pathname of the child for which to get relative name.

    Returns
    -------
    dict
        Maps relative name in parent (owner of unknowns and unknowns_dict) to
        the corresponding relative name in the child, where relative name may
        include the 'promoted' name of a variable.
    """
    # unknowns is keyed on name relative to the parent system
    # unknowns_dict is keyed on absolute pathname
    umap = {}
    for rel, meta in unknowns.items():
        abspath = meta['pathname']
        if abspath.startswith(child_name+'.'):
            newmeta = unknowns_dict.get(abspath)
            if newmeta is not None:
                newrel = newmeta['promoted_name']
            else:
                newrel = rel
            umap[rel] = newrel

    return umap

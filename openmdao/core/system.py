""" Base class for all systems in OpenMDAO."""

from collections import OrderedDict
from fnmatch import fnmatch
from itertools import chain
from six import string_types, iteritems

import numpy as np

from openmdao.core.mpiwrap import MPI, get_comm_if_active
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

        Parameters
        ----------
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

        Parameters
        ----------
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
                    rel = meta.get('relative_name', n)
                    if rel == name:
                        return True

        return False

    def subsystems(self):
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
            self.pathname = ':'.join((parent_path, self.name))
        else:
            self.pathname = self.name

    def preconditioner(self):
        pass

    def jacobian(self, params, unknowns, resids):
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        raise NotImplementedError("solve_nonlinear")

    def apply_nonlinear(self, params, unknowns, resids):
        pass

    def solve_linear(self, rhs, params, unknowns, mode="fwd"):
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

        Parameters
        ----------
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.
        """
        self.comm = get_comm_if_active(self, comm)

    def _set_vars_as_remote(self):
        """
        Set 'remote' attribute in metadata of all variables for this subsystem.
        """
        pname = self.pathname + ':'
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

        Parameters
        ----------
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
                    states.append(meta['relative_name'])
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
            if hasattr(self, '_varmanager'):
                param_src = self._varmanager.connections.get(p_name)
                if param_src is not None:

                    # Have to convert to relative name to key into unknowns
                    if param_src not in self.unknowns:
                        for name in unknowns:
                            meta = unknowns.metadata(name)
                            if meta['pathname'] == param_src:
                                param_src = meta['relative_name']

                    target_input = unknowns.flat[param_src]

            mydict = {}
            for key, val in self._params_dict.items():
                if val['relative_name'] == p_name:
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
                dresids[unknown] += J.dot(arg_vec[param].flatten()).reshape(result.shape)
            else:
                arg_vec[param] += J.T.dot(result.flatten()).reshape(arg_vec[param].shape)

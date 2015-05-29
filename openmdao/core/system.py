""" Base class for all systems in OpenMDAO."""

from collections import OrderedDict
import copy
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
                                   desc = "Set to True to finite difference this system.")
        self.fd_options.add_option('form', 'forward',
                                   values = ['forward', 'backward', 'central', 'complex_step'],
                                   desc = "Finite difference mode. (forward, backward, central) "
                                   "You can also set to 'complex_step' to peform the complex "
                                   "step method if your components support it.")
        self.fd_options.add_option("step_size", 1.0e-6,
                                    desc = "Default finite difference stepsize")
        self.fd_options.add_option("step_type", 'absolute',
                                   values = ['absolute', 'relative'],
                                   desc = 'Set to absolute, relative')

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

    def fd_jacobian(self, params, unknowns, resids, step_size=None, form=None,
                    step_type=None):
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

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """

        # Function call arguments have precedence over the system dict.
        if step_size == None:
            step_size = self.fd_options['step_size']
        if form == None:
            form = self.fd_options['form']
        if step_type == None:
            step_type = self.fd_options['step_type']

        jac = {}
        resid_cache = resids.vec.copy()
        resid_cache2 = None

        states = []
        for u_name, meta in iteritems(self._unknowns_dict):
            if meta.get('state'):
                states.append(meta['relative_name'])

        # Compute gradient for this param or state.
        for p_name in chain(params, states):

            if p_name in states:
                inputs = unknowns
            else:
                inputs = params

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
            for u_name in unknowns:
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

                    inputs.flat[p_name][idx] += step

                    self.apply_nonlinear(params, unknowns, resids)

                    inputs.flat[p_name][idx] -= step

                    # delta resid is delta unknown
                    resids.vec[:] -= resid_cache
                    resids.vec[:] *= (1.0/step)

                elif fdform == 'backward':

                    inputs.flat[p_name][idx] -= step

                    self.apply_nonlinear(params, unknowns, resids)

                    inputs.flat[p_name][idx] += step

                    # delta resid is delta unknown
                    resids.vec[:] -= resid_cache
                    resids.vec[:] *= (-1.0/step)

                elif fdform == 'central':

                    inputs.flat[p_name][idx] += step
                    self.apply_nonlinear(params, unknowns, resids)
                    resids2 = resids.vec - resid_cache

                    resids.vec[:] = resid_cache

                    inputs.flat[p_name][idx] -= 2.0*step
                    self.apply_nonlinear(params, unknowns, resids)

                    # central difference formula
                    resids.vec[:] -= resid_cache + resids2
                    resids.vec[:] *= (-0.5/step)

                    inputs.flat[p_name][idx] += step

                for u_name in unknowns:
                    jac[u_name, p_name][:, idx] = resids.flat[u_name]

                # Restore old residual
                resids.vec[:] = resid_cache

        return jac

""" Base class for all systems in OpenMDAO."""

import copy
from six import string_types, iteritems
from collections import OrderedDict
from fnmatch import fnmatch
from itertools import chain

import numpy as np

from openmdao.core.mpiwrap import MPI, FakeComm, get_comm_if_active
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

        self.comm = FakeComm()

        self.fd_options = OptionsDictionary()
        self.fd_options.add_option('force_fd', False,
                                   doc = "Set to True to finite difference this system.")
        self.fd_options.add_option('form', 'forward',
                                   values = ['forward', 'backward', 'central', 'complex_step'],
                                   doc = "Finite difference mode. (forward, backward, central) "
                                   "You can also set to 'complex_step' to peform the complex "
                                   "step method if your components support it.")
        self.fd_options.add_option("step", 1.0e-6,
                                    doc = "Default finite difference stepsize")
        self.fd_options.add_option("step_type", 'absolute',
                                   values = ['absolute', 'relative', 'bounds_scaled'],
                                   doc = 'Set to absolute, relative, '
                                   'or scaled to the bounds (high-low) step sizes')

    def __getitem__(self, name):
        """
        Return the variable or subsystem of the given name from this system.

        Parameters
        ----------
        name : str
            the name of the variable or subsystem

        Returns
        -------
        value OR `System`
            the unflattened value of the given variable OR a reference to
            the named `System`
        """
        raise RuntimeError("Variable '%s' must be accessed from a containing Group" % name)


    def promoted(self, name):
        """Determine is the given variable name  is being promoted from this
        `System`.

        Parameters
        ----------
        name : str
            the name of a variable, relative to this `System`

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
            the pathname of the parent `System`, which is to be prepended to the
            name of this child `System`
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
        pass

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
            processors usable by this `System`
        """
        return (1, 1)

    def _setup_communicators(self, comm):
        """
        Assign communicator to this `System` and all of it's subsystems

        Parameters
        ----------
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.
        """
        self.comm = get_comm_if_active(self, comm)

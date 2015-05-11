from six import string_types
from collections import OrderedDict
from fnmatch import fnmatch
from itertools import chain

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
        raise RuntimeError('Variables must be accessed from a containing Group')

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
            raise TypeError("%s promotes must be specified as a list, "
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

    def jacobian(self, params, unknowns):
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        pass

    def solve_linear(self, rhs, params, unknowns, resids, dparams, dunknowns, dresids, mode="fwd"):
        pass

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode="fwd"):
        pass

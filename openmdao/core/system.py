from collections import OrderedDict


class System(object):
    """ Base class for systems in OpenMDAO."""

    def __init__(self):
        self.name = ''
        self.pathname = ''

        self._params_dict = OrderedDict()
        self._unknowns_dict = OrderedDict()

        # by default, don't promote any vars up to our parent
        self.promotes = ()

        self.ln_solver = None
        self.nl_solver = None

    def promoted(self, name):
        # TODO: handle wildcards
        return name in self.promotes

    def _setup_paths(self, parent_path):
        """Set the absolute pathname of each System in the
        tree.
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

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns, dresids, mode="fwd"):
        pass

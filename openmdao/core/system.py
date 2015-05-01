""" Base class for systems in OpenMDAO."""


class System(object):
    def __init__(self):
        self.name = ''
        self.pathname = ''
        # by default, don't promote any vars up to our parent
        self.promotes = ()

    def promoted(self, name):
        # TODO: handle wildcards
        return name in self.promotes

    def setup_syspaths(self, parent_path):
        """Set the absolute pathname of each System in the
        tree.
        """
        if parent_path:
            self.pathname = ':'.join((parent_path, self.name))
        else:
            self.pathname = self.name

    def setup_vectors(self, parent_vm=None):
        pass

    def preconditioner(self):
        pass

    def jacobian(self, params, unknowns):
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        pass

    def solve_linear(self, rhs, params, unknowns, resids, dparams, dunknowns,
        dresids, mode="fwd"):
        pass

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns,
        dresids, mode="fwd"):
        pass

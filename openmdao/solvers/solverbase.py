""" Base class for linear and nonlinear solvers."""


class LinearSolver(object):
    """ Base class for all linear solvers. Inherit from this class to create a
    new custom linear solver."""

    def __init__(self):
        self.iter_count = 0

    def solve(self, rhs):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        rhs: ndarray
            Array containing the right hand side for the linear solve. Also
            possibly a 2D array with multiple right hand sides.
        """
        pass


class NonLinearSolver(object):
    """ Base class for all nonlinear solvers. Inherit from this class to create a
    new custom nonlinear solver."""

    def __init__(self):
        self.iter_count = 0

    def solve(self):
        """ Drive all residuals in self.system and all subsystems to zero.
        This includes all implicit components.
        """
        pass



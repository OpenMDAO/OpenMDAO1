""" Base class for linear and nonlinear solvers."""

from openmdao.core.options import OptionsDictionary


class LinearSolver(object):
    """ Base class for all linear solvers. Inherit from this class to create a
    new custom linear solver."""

    def __init__(self):
        self.iter_count = 0
        self.options = OptionsDictionary()

    def solve(self, params, unknowns, resids, system):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned. This function must be defined
        when inheriting.

        Parameters
        ----------
        rhs : ndarray
            Array containing the right hand side for the linear solve. Also
            possibly a 2D array with multiple right hand sides.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'

        Returns
        -------
        ndarray : Solution vector
        """
        pass


class NonLinearSolver(object):
    """ Base class for all nonlinear solvers. Inherit from this class to create a
    new custom nonlinear solver."""

    def __init__(self):
        self.iter_count = 0
        self.options = OptionsDictionary()

    def solve(self):
        """ Drive all residuals in self.system and all subsystems to zero.
        This includes all implicit components. This function must be defined
        when inheriting.

        Parameters
        ----------
        params : `VecWrapper`
            `VecWrapper` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.
        """
        pass



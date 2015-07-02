""" Base class for linear and nonlinear solvers."""

from openmdao.core.options import OptionsDictionary


class LinearSolver(object):
    """ Base class for all linear solvers. Inherit from this class to create a
    new custom linear solver."""

    def __init__(self):
        self.iter_count = 0
        self.options = OptionsDictionary()
        self.recorders = []

    def add_recorder(self, recorder):
        """Appends the given recorder to this solver's list of recorders.

        Args
        ----
        recorder: `BaseRecorder`
            A recorder object.
        """
        self.recorders.append(recorder)

    def solve(self, rhs, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned. This function must be defined
        when inheriting.

        Args
        ----
        rhs : ndarray
            Array containing the right-hand side for the linear solve. Also
            possibly a 2D array with multiple right-hand sides.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

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
        self.recorders = []

    def add_recorder(self, recorder):
        """Appends the given recorder to this solver's list of recorders.

        Args
        ----
        recorder: `BaseRecorder`
            A recorder object.
        """
        self.recorders.append(recorder)

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Drive all residuals in self.system and all subsystems to zero.
        This includes all implicit components. This function must be defined
        when inheriting.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        pass

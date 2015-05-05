""" Gauss Seidel non-linear solver."""

from openmdao.solvers.solverbase import NonLinearSolver


class NL_Gauss_Seidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for an
    OpenMDAO group. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.
    """

    def solve(self, params, unknowns, resids):
        """ Solves the system using Gauss Seidel.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)

        resids: vecwrapper
            VecWrapper containing residuals. (r)
        """
        pass

""" Gauss Seidel non-linear solver."""

from openmdao.solvers.solverbase import NonLinearSolver


class NLGaussSeidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for an
    OpenMDAO group. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.
    """

    def solve(self, params, unknowns, resids, system):
        """ Solves the system using Gauss Seidel.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)

        resids: vecwrapper
            VecWrapper containing residuals. (r)

        system: system
            Parent system object.
        """
        self.iter_count = 0
        #atol = self.options['atol']
        #rtol = self.options['rtol']
        #maxiter = self.options['maxiter']

        system.children_solve_nonlinear()

        # TODO - turn into Gauss Seidel as follows

        varmanager = system._varmanager
        resids = varmanager.resids
        #normval = resids.norm()

        # while self.iter_count < maxiter and normval > self.atol

            #"""Runs an iteration."""
            #self.iter_count += 1
            #self.children_solve_nonlinear()

            #normval = resids.norm()


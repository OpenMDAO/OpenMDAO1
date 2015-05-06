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

        system.children_solve_nonlinear()

        # TODO - turn into Gauss Seidel as follows

        #self.children_solve_nonlinear()
        #self.normval = get_norm(self.workflow._system.vec['f'],
                                #self._norm_order)
        #self.norm0 = self.normval if self.normval != 0.0 else 1.0

        # while self.iter_count < self.maxiter and \
               #self.normval > self.tolerance

            #"""Runs an iteration."""
            #self.iter_count += 1
            #uvec = system.vec['u']
            #fvec = system.vec['f']
            #self.children_solve_nonlinear()

            #self.normval = get_norm(self.workflow._system.vec['f'],
                                    #self._norm_order)


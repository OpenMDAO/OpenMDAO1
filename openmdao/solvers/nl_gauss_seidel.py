""" Gauss Seidel non-linear solver."""

from openmdao.solvers.solverbase import NonLinearSolver


class NLGaussSeidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for an
    OpenMDAO group. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.
    """

    def __init__(self):
        super(NLGaussSeidel, self).__init__()

        self.options.add_option('atol', 1e-6)
        self.options.add_option('rtol', 1e-6)
        self.options.add_option('maxiter', 100)


    def solve(self, params, unknowns, resids, system):
        """ Solves the system using Gauss Seidel.

        Parameters
        ----------
        params: `VecWrapper`
            `VecWrapper` containing parameters (p)

        unknowns: `VecWrapper`
            `VecWrapper` containing outputs and states (u)

        resids: `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system: `System`
            Parent `System` object.
        """

        atol = self.options['atol']
        rtol = self.options['rtol']
        maxiter = self.options['maxiter']

        self.iter_count = 1
        system.children_solve_nonlinear()

        # Bail early if the user wants to.
        if maxiter == 1:
            return

        varmanager = system._varmanager
        resids = varmanager.resids

        system.apply_nonlinear(params, unknowns, resids)
        normval = resids.norm()
        basenorm = normval if normval > atol else 1.0

        while self.iter_count < maxiter and \
              normval > atol and \
              normval/basenorm > rtol:

            # Runs an iteration
            self.iter_count += 1
            system.children_solve_nonlinear()

            system.apply_nonlinear(params, unknowns, resids)
            normval = resids.norm()


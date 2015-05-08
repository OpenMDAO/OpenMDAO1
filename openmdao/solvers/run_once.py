""" The RunOnce solver just performs solve_nonlinear the system hierarchy
with no iteration.."""

from openmdao.solvers.solverbase import NonLinearSolver


class RunOnce(NonLinearSolver):
    """ The RunOnce solver just performs solve_nonlinear the system hierarchy
    with no iteration..
    """

    def __init__(self):
        super(RunOnce, self).__init__()

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
        system.children_solve_nonlinear()
        self.iter_count += 1

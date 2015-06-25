""" The RunOnce solver just performs solve_nonlinear on the system hierarchy
with no iteration."""

from openmdao.solvers.solverbase import NonLinearSolver


class RunOnce(NonLinearSolver):
    """ The RunOnce solver just performs solve_nonlinear on the system hierarchy
    with no iteration.
    """

    def __init__(self):
        super(RunOnce, self).__init__()

    def solve(self, params, unknowns, resids, system):
        """ Executes each item in the system hierarchy sequentially.

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
        """
        #TODO: When to record?
        system.children_solve_nonlinear()
        self.iter_count += 1

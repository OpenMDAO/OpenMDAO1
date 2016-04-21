""" The RunOnce solver just performs solve_nonlinear on the system hierarchy
with no iteration."""

from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import create_local_meta, update_local_meta


class RunOnce(NonLinearSolver):
    """ The RunOnce solver just performs solve_nonlinear on the system hierarchy
    with no iteration.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout,
        set to 2 to print subiteration residuals as well.

    """

    def __init__(self):
        super(RunOnce, self).__init__()
        self.options.remove_option('err_on_maxiter')
        self.print_name = 'RUN_ONCE'

    def solve(self, params, unknowns, resids, system, metadata=None):
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

        self.iter_count += 1
        # Metadata setup
        local_meta = create_local_meta(metadata, system.name)
        system.ln_solver.local_meta = local_meta
        update_local_meta(local_meta, (self.iter_count,))

        system.children_solve_nonlinear(local_meta)
        self.recorders.record_iteration(system, local_meta)

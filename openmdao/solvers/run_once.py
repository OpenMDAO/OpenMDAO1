""" The RunOnce solver just performs solve_nonlinear on the system hierarchy
with no iteration."""

from openmdao.solvers.solverbase import NonLinearSolver
from openmdao.util.recordutil import create_local_meta, update_local_meta


class RunOnce(NonLinearSolver):
    """ The RunOnce solver just performs solve_nonlinear on the system hierarchy
    with no iteration.
    """

    def __init__(self):
        super(RunOnce, self).__init__()

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
        if metadata is None:
            parent_coordinate = []
            parent_iteration = {}
        else:
            parent_coordinate, parent_iteration = metadata['current']

        self.iter_count += 1
        # Metadata setup
        local_meta = create_local_meta(metadata, system.name)
        update_local_meta(local_meta, (self.iter_count,))

        system.children_solve_nonlinear(local_meta)
        for recorder in self.recorders:
            recorder.raw_record(system.params, system.unknowns, system.resids, local_meta)


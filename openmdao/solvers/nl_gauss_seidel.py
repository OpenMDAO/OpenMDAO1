""" Gauss Seidel non-linear solver."""

from openmdao.solvers.solverbase import NonLinearSolver
from openmdao.util.recordutil import update_local_meta, create_local_meta


class NLGaussSeidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for a
    `Group`. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.
    """

    def __init__(self):
        super(NLGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-6,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 100,
                       desc='Maximum number of iterations.')

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using Gauss Seidel.

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

        atol = self.options['atol']
        rtol = self.options['rtol']
        maxiter = self.options['maxiter']

        # Initial run
        self.iter_count = 1

        # Metadata setup
        local_meta = create_local_meta(metadata, system.name)
        update_local_meta(local_meta, (self.iter_count,))

        # Initial Solve
        system.children_solve_nonlinear(local_meta)

        for recorder in self.recorders:
            recorder.raw_record(params, unknowns, resids, local_meta)

        # Bail early if the user wants to.
        if maxiter == 1:
            return

        resids = system.resids

        # Evaluate Norm
        system.apply_nonlinear(params, unknowns, resids)
        normval = resids.norm()
        basenorm = normval if normval > atol else 1.0

        while self.iter_count < maxiter and \
                normval > atol and \
                normval/basenorm > rtol:

            # Metadata update
            self.iter_count += 1
            update_local_meta(local_meta, (self.iter_count,))

            # Runs an iteration
            system.children_solve_nonlinear(local_meta)
            for recorder in self.recorders:
                recorder.raw_record(params, unknowns, resids, local_meta)

            # Evaluate Norm
            system.apply_nonlinear(params, unknowns, resids)
            normval = resids.norm()

""" Backracking line search using the Armijo–Goldstein condition."""

import numpy as np

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import LineSearch
from openmdao.util.record_util import update_local_meta, create_local_meta


class BackTracking(LineSearch):
    """A line search subsolver that implements backracking using the
    Armijo–Goldstein condition..

    Options
    -------
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout
        each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(10)
        Maximum number of line searches.
    options['solve_subsystems'] :  bool(True)
        Set to True to solve subsystems. You may need this for solvers nested under Newton.
    """

    def __init__(self):
        super(BackTracking, self).__init__()

        opt = self.options
        opt.add_option('maxiter', 0, lower=0,
                       desc='Maximum number of line searches.')
        opt.add_option('solve_subsystems', True,
                       desc='Set to True to solve subsystems. You may need this for solvers nested under Newton.')

        self.print_name = 'BK_TKG'

    def solve(self, params, unknowns, resids, system, solver, alpha, fnorm0,
              metadata=None):
        """ Take the gradient calculated by the parent solver and figure out
        how far to go.

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

        solver : `Solver`
            Parent solver instance.

        alpha : ndarray
            Initial over-relaxation factors as used in parent solver.

        fnorm0 : float
            Initial norm of the residual for iteration printing.

        Returns
        --------
        float
            Norm of the final residual
        """

        maxiter = self.options['maxiter']
        result = system.dumat[None]
        local_meta = create_local_meta(metadata, system.pathname)

        # Initial execution really belongs to our parent driver's iteration,
        # so use its info.
        fnorm = resids.norm()

        itercount = 0
        ls_alpha = alpha

        # Further backtacking if needed.
        while itercount < maxiter:

            ls_alpha *= 0.5
            unknowns.vec -= ls_alpha*result.vec
            itercount += 1

            # Metadata update
            update_local_meta(local_meta, (solver.iter_count, itercount))

            # Just evaluate the model with the new points
            if self.options['solve_subsystems']:
                system.children_solve_nonlinear(local_meta)
            system.apply_nonlinear(params, unknowns, resids, local_meta)

            solver.recorders.record_iteration(system, local_meta)

            fnorm = resids.norm()
            if self.options['iprint'] > 0:
                self.print_norm(self.print_name, system.pathname, itercount,
                                fnorm, fnorm0, indent=1, solver='LS')

        if itercount >= maxiter and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': BackTracking failed to converge after %d "
                                "iterations." % (system.pathname, maxiter))

        return fnorm

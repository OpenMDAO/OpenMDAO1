""" Backtracking line search using the Armijo-Goldstein condition."""

from math import isnan

import numpy as np

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import LineSearch
from openmdao.util.record_util import update_local_meta, create_local_meta


class BackTracking(LineSearch):
    """A line search subsolver that implements backracking using the
    Armijo-Goldstein condition..

    Options
    -------
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout
    options['maxiter'] :  int(10)
        Maximum number of line searches.
    options['solve_subsystems'] :  bool(True)
        Set to True to solve subsystems. You may need this for solvers nested under Newton.
    options['rho'] : int(0.5)
        Backtracking step.
    options['c'] : int(0.5)
        Slope check trigger.
    """

    def __init__(self):
        super(BackTracking, self).__init__()

        opt = self.options
        opt.add_option('maxiter', 5, lower=0,
                       desc='Maximum number of line searches.')
        opt.add_option('solve_subsystems', True,
                       desc='Set to True to solve subsystems. You may need this for solvers nested under Newton.')
        opt.add_option('rho', 0.5,
                       desc="Backtracking step.")
        opt.add_option('c', 0.5,
                       desc="Slope check trigger.")

        self.print_name = 'BK_TKG'

    def solve(self, params, unknowns, resids, system, solver, alpha_scalar, alpha,
              base_u, base_norm, fnorm, fnorm0, metadata=None):
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

        solver : `Solver`
            Parent solver instance.

        alpha_scalar : float
            Initial over-relaxation factor as used in parent solver.

        alpha : ndarray
            Initial over-relaxation factor as used in parent solver, vector
            (so we don't re-allocate).

        base_u : ndarray
            Initial value of unknowns before the Newton step.

        base_norm : float
            Norm of the residual prior to taking the Newton step.

        fnorm : float
            Norm of the residual after taking the Newton step.

        fnorm0 : float
            Initial norm of the residual for iteration printing.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).

        Returns
        --------
        float
            Norm of the final residual
        """

        maxiter = self.options['maxiter']
        rho = self.options['rho']
        c = self.options['c']
        result = system.dumat[None]
        local_meta = create_local_meta(metadata, system.pathname)

        itercount = 0
        ls_alpha = alpha_scalar

        # Further backtacking if needed.
        # The Armijo-Goldstein is basically a slope comparison --actual vs predicted.
        # We don't have an actual gradient, but we have the Newton vector that should
        # take us to zero, and our "runs" are the same, and we can just compare the
        # "rise".
        while itercount < maxiter and (base_norm - fnorm) < c*ls_alpha*base_norm:

            ls_alpha *= rho

            # If our step will violate any upper or lower bounds, then reduce
            # alpha in just that direction so that we only step to that
            # boundary.
            unknowns.vec[:] = base_u
            alpha[:] = ls_alpha
            alpha = unknowns.distance_along_vector_to_limit(alpha, result)

            unknowns.vec += alpha*result.vec
            itercount += 1

            # Metadata update
            update_local_meta(local_meta, (solver.iter_count, itercount))

            # Just evaluate the model with the new points
            if self.options['solve_subsystems']:
                system.children_solve_nonlinear(local_meta)
            system.apply_nonlinear(params, unknowns, resids, local_meta)

            solver.recorders.record_iteration(system, local_meta)

            fnorm = resids.norm()
            if self.options['iprint'] == 2:
                self.print_norm(self.print_name, system.pathname, itercount,
                                fnorm, fnorm0, indent=1, solver='LS')


        # Final residual print if you only want the last one
        if self.options['iprint'] == 1:
            self.print_norm(self.print_name, system.pathname, itercount,
                            fnorm, fnorm0, indent=1, solver='LS')

        if itercount >= maxiter or isnan(fnorm):

            if self.options['err_on_maxiter']:
                msg = "Solve in '{}': BackTracking failed to converge after {} " \
                      "iterations."
                raise AnalysisError(msg.format(system.pathname, maxiter))

            msg = 'FAILED to converge after %d iterations' % itercount
            fail = True
        else:
            msg = 'Converged in %d iterations' % itercount
            fail = False

        if self.options['iprint'] > 0 or fail:

            self.print_norm(self.print_name, system.pathname, itercount,
                            fnorm, fnorm0, msg=msg, indent=1, solver='LS')

        return fnorm

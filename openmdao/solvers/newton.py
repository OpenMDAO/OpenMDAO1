""" Non-linear solver that implements a Newton's method."""

from math import isnan

import numpy as np

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class Newton(NonLinearSolver):
    """A python Newton solver that solves a linear system to determine the
    next direction to step. Also uses `Backtracking` as the default line
    search algorithm, but you can choose a different one by specifying
    `self.line_search`. A linear solver can also be specified by assigning it
    to `self.ln_solver` to use a different solver than the one in the
    parent system.

    Options
    -------
    options['alpha'] :  float(1.0)
        Initial over-relaxation factor.
    options['atol'] :  float(1e-12)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout
        each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(20)
        Maximum number of iterations.
    options['rtol'] :  float(1e-10)
        Relative convergence tolerance.
    options['solve_subsystems'] :  bool(True)
        Set to True to solve subsystems. You may need this for solvers nested under Newton.
    """

    def __init__(self):
        super(Newton, self).__init__()

        # What we support
        self.supports['uses_derivatives'] = True

        opt = self.options
        opt.add_option('atol', 1e-12, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-10, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 20, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('alpha', 1.0,
                       desc='Initial over-relaxation factor.')
        opt.add_option('solve_subsystems', True,
                       desc='Set to True to solve subsystems. You may need this for solvers nested under Newton.')

        self.print_name = 'NEWTON'

        # User can optionally specify a line search.
        self.line_search = None

        # User can specify a different linear solver for Newton. Default is
        # to use the parent's solver.
        self.ln_solver = None

    def setup(self, sub):
        """ Initialize sub solvers.

        Args
        ----
        sub: `System`
            System that owns this solver.
        """
        if self.line_search:
            self.line_search.setup(sub)
        if self.ln_solver:
            self.ln_solver.setup(sub)

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Solves the system using a Netwon's Method.

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
        alpha_scalar = self.options['alpha']
        ls = self.line_search

        # Metadata setup
        self.iter_count = 0
        local_meta = create_local_meta(metadata, system.pathname)
        if self.ln_solver:
            self.ln_solver.local_meta = local_meta
        else:
            system.ln_solver.local_meta = local_meta
        update_local_meta(local_meta, (self.iter_count, 0))

        # Perform an initial run to propagate srcs to targets.
        system.children_solve_nonlinear(local_meta)
        system.apply_nonlinear(params, unknowns, resids)

        if ls:
            base_u = np.zeros(unknowns.vec.shape)

        f_norm = resids.norm()
        f_norm0 = f_norm

        if self.options['iprint'] > 0:
            self.print_norm(self.print_name, system.pathname, 0, f_norm,
                            f_norm0)

        arg = system.drmat[None]
        result = system.dumat[None]

        while self.iter_count < maxiter and f_norm > atol and \
                f_norm/f_norm0 > rtol:

            # Linearize Model with partial derivatives
            system._sys_linearize(params, unknowns, resids, total_derivs=False)

            # Calculate direction to take step
            arg.vec[:] = -resids.vec
            with system._dircontext:
                system.solve_linear(system.dumat, system.drmat,
                                    [None], mode='fwd', solver=self.ln_solver)

            self.iter_count += 1

            # Allow different alphas for each value so we can keep moving when we
            # hit a bound.
            alpha = alpha_scalar*np.ones(len(unknowns.vec))

            # If our step will violate any upper or lower bounds, then reduce
            # alpha in just that direction so that we only step to that
            # boundary.
            alpha = unknowns.distance_along_vector_to_limit(alpha, result)

            # Cache the current norm
            if ls:
                base_u[:] = unknowns.vec
                base_norm = f_norm

            # Apply step that doesn't violate bounds
            unknowns.vec += alpha*result.vec

            # Metadata update
            update_local_meta(local_meta, (self.iter_count, 0))

            # Just evaluate (and optionally solve) the model with the new
            # points
            if self.options['solve_subsystems']:
                system.children_solve_nonlinear(local_meta)
            system.apply_nonlinear(params, unknowns, resids, local_meta)

            self.recorders.record_iteration(system, local_meta)

            f_norm = resids.norm()
            if self.options['iprint'] > 0:
                self.print_norm(self.print_name, system.pathname, self.iter_count,
                                f_norm, f_norm0)

            # Line Search to determine how far to step in the Newton direction
            if ls:
                f_norm = ls.solve(params, unknowns, resids, system, self,
                                  alpha_scalar, alpha, base_u, base_norm,
                                  f_norm, f_norm0, metadata)


        # Need to make sure the whole workflow is executed at the final
        # point, not just evaluated.
        #self.iter_count += 1
        #update_local_meta(local_meta, (self.iter_count, 0))
        #system.children_solve_nonlinear(local_meta)

        if self.iter_count >= maxiter or isnan(f_norm):
            msg = 'FAILED to converge after %d iterations' % self.iter_count
            fail = True
        else:
            fail = False

        if self.options['iprint'] > 0:

            if not fail:
                msg = 'converged'

            self.print_norm(self.print_name, system.pathname, self.iter_count,
                            f_norm, f_norm0, msg=msg)

        if fail and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': Newton %s" % (system.pathname,
                                                              msg))

    def print_all_convergence(self):
        """ Turns on iprint for this solver and all subsolvers. Override if
        your solver has subsolvers."""
        self.options['iprint'] = 1
        if self.line_search:
            self.line_search.options['iprint'] = 1

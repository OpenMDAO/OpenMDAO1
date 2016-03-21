""" Gauss Seidel non-linear solver."""

from math import isnan

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import NonLinearSolver
from openmdao.util.record_util import update_local_meta, create_local_meta


class NLGaussSeidel(NonLinearSolver):
    """ Nonlinear Gauss Seidel solver. This is the default solver for a
    `Group`. If there are no cycles, then the system will solve its
    subsystems once and terminate. Equivalent to fixed point iteration in
    cases with cycles.

    Options
    -------
    options['atol'] :  float(1e-06)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(100)
        Maximum number of iterations.
    options['rtol'] :  float(1e-06)
        Relative convergence tolerance.

    """

    def __init__(self):
        super(NLGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-6, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 100, lower=0,
                       desc='Maximum number of iterations.')

        self.print_name = 'NLN_GS'

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
        iprint = self.options['iprint']

        # Initial run
        self.iter_count = 1

        # Metadata setup
        local_meta = create_local_meta(metadata, system.pathname)
        system.ln_solver.local_meta = local_meta
        update_local_meta(local_meta, (self.iter_count,))

        # Initial Solve
        system.children_solve_nonlinear(local_meta)

        self.recorders.record_iteration(system, local_meta)

        # Bail early if the user wants to.
        if maxiter == 1:
            return

        resids = system.resids

        # Evaluate Norm
        system.apply_nonlinear(params, unknowns, resids)
        normval = resids.norm()
        basenorm = normval if normval > atol else 1.0

        if self.options['iprint'] > 0:
            self.print_norm(self.print_name, system.pathname, 0, normval, basenorm)

        while self.iter_count < maxiter and \
                normval > atol and \
                normval/basenorm > rtol:

            # Metadata update
            self.iter_count += 1
            update_local_meta(local_meta, (self.iter_count,))

            # Runs an iteration
            system.children_solve_nonlinear(local_meta)
            self.recorders.record_iteration(system, local_meta)

            # Evaluate Norm
            system.apply_nonlinear(params, unknowns, resids)
            normval = resids.norm()

            if self.options['iprint'] > 0:
                self.print_norm(self.print_name, system.pathname, self.iter_count, normval,
                                basenorm)

        if self.iter_count >= maxiter or isnan(normval):
            msg = 'FAILED to converge after %d iterations' % self.iter_count
            fail = True
        else:
            fail = False

        if self.options['iprint'] > 0:
            if not fail:
                msg = 'converged'

            self.print_norm(self.print_name, system.pathname, self.iter_count, normval,
                            basenorm, msg=msg)

        if fail and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': NLGaussSeidel %s" %
                                (system.pathname, msg))

""" Gauss Seidel non-linear solver."""

from math import isnan

import numpy as np

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import error_wrap_nl, NonLinearSolver
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
        Set to 0 to print only failures, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout,
        or -1 to suppress all printing.
    options['maxiter'] :  int(100)
        Maximum number of iterations.
    options['rtol'] :  float(1e-06)
        Relative convergence tolerance.
    options['utol'] :  float(1e-12)
        Convergence tolerance on the change in the unknowns.
    options['use_aitken'] : bool(False)
        Set to True to use Aitken acceleration.
    options['aitken_alpha_min'] : float(0.25)
        Lower limit for Aitken relaxation factor.
    options['aitken_alpha_max'] : float(2.0)
        Upper limit for Aitken relaxation factor.

    """

    def __init__(self):
        super(NLGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-6, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-6, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('utol', 1e-12, lower=0.0,
                       desc='Convergence tolerance on the change in the unknowns.')
        opt.add_option('maxiter', 100, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('use_aitken', False,
                       desc='Set to True to use Aitken acceleration.')
        opt.add_option('aitken_alpha_min', 0.25,
                       desc='Lower limit for Aitken relaxation factor.')
        opt.add_option('aitken_alpha_max', 2.0,
                       desc='Upper limit for Aitken relaxation factor.')

        self.print_name = 'NLN_GS'
        self.delta_u_n_1 = 'None' # delta_u_n-1 for Aitken acc.
        self.aitken_alpha = 1.0 # Initial Aitken relaxation factor 

    def setup(self, sub):
        """ Initialize this solver.

        Args
        ----
        sub: `System`
            System that owns this solver.
        """
        if sub.is_active():
            self.unknowns_cache = np.empty(sub.unknowns.vec.shape)

    @error_wrap_nl
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
        utol = self.options['utol']
        maxiter = self.options['maxiter']
        iprint = self.options['iprint']
        unknowns_cache = self.unknowns_cache

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
        unknowns_cache = np.zeros(unknowns.vec.shape)

        # Evaluate Norm
        system.apply_nonlinear(params, unknowns, resids)
        normval = resids.norm()
        basenorm = normval if normval > atol else 1.0
        u_norm = 1.0e99

        if iprint == 2:
            self.print_norm(self.print_name, system, 1, normval, basenorm)

        while self.iter_count < maxiter and \
                normval > atol and \
                normval/basenorm > rtol  and \
                u_norm > utol:

            # Metadata update
            self.iter_count += 1
            update_local_meta(local_meta, (self.iter_count,))
            unknowns_cache[:] = unknowns.vec

            # Runs an iteration
            system.children_solve_nonlinear(local_meta)
            self.recorders.record_iteration(system, local_meta)

            # Evaluate Norm
            system.apply_nonlinear(params, unknowns, resids)
            normval = resids.norm()
            u_norm = np.linalg.norm(unknowns.vec - unknowns_cache)

            if self.options['use_aitken']: # If Aitken acceleration is enabled
                
                # This method is used by Kenway et al. in "Scalable Parallel  
                # Approach for High-Fidelity Steady-State Aeroelastic Analysis 
                # and Adjoint Derivative Computations" (line 22 of Algorithm 1)
                # It is based on "A version of the Aitken accelerator for 
                # computer iteration" by Irons et al. 
            
                # Use relaxation after second iteration
                # self.delta_u_n_1 is a string for the first iteration
                if (type(self.delta_u_n_1) is not str) and \
                    normval > atol and \
                    normval/basenorm > rtol  and \
                    u_norm > utol:

                    delta_u_n = unknowns.vec - unknowns_cache
                    delta_u_n_1 = self.delta_u_n_1

                    # Compute relaxation factor 
                    self.aitken_alpha = self.aitken_alpha * \
                        (1. - np.dot((delta_u_n  - delta_u_n_1), delta_u_n) \
                        / np.linalg.norm((delta_u_n  - delta_u_n_1), 2)**2)

                    # Limit relaxation factor to desired range
                    self.aitken_alpha = max(self.options['aitken_alpha_min'], 
                        min(self.options['aitken_alpha_max'], self.aitken_alpha))

                    if iprint == 1 or iprint == 2:
                        print("Aitken relaxation factor is", self.aitken_alpha)

                    self.delta_u_n_1 = delta_u_n.copy()

                    # Update unknowns vector
                    unknowns.vec[:] = unknowns_cache + self.aitken_alpha * delta_u_n

                elif (type(self.delta_u_n_1) is str): # For the first iteration
                    # Initially self.delta_u_n_1 is a string then it is replaced
                    # by the following vector
                    self.delta_u_n_1 = unknowns.vec - unknowns_cache 

            if iprint == 2:
                self.print_norm(self.print_name, system, self.iter_count, normval,
                                basenorm, u_norm=u_norm)

        # Final residual print if you only want the last one
        if iprint == 1:
            self.print_norm(self.print_name, system, self.iter_count, normval,
                            basenorm, u_norm=u_norm)

        if self.iter_count >= maxiter or isnan(normval):
            msg = 'FAILED to converge after %d iterations' % self.iter_count
            fail = True
        else:
            fail = False

        if iprint > 0 or (fail and iprint > -1 ):
            if not fail:
                msg = 'Converged in %d iterations' % self.iter_count

            self.print_norm(self.print_name, system, self.iter_count, normval,
                            basenorm, msg=msg)

        if fail and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': NLGaussSeidel %s" %
                                (system.pathname, msg))

""" Gauss Seidel non-linear solver that implements a Newton's method."""

from __future__ import print_function

from openmdao.solvers.solverbase import NonLinearSolver


class Newton(NonLinearSolver):
    """A python Newton solver with line-search adapation of the relaxation
    parameter.
    """

    def __init__(self):
        super(Newton, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-10,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 20,
                       desc='Maximum number of iterations.')
        opt.add_option('ls_atol', 1e-10,
                       desc='Absolute convergence tolerancee for line search.')
        opt.add_option('ls_rtol', 0.9,
                       desc='Relative convergence tolerancee for line search.')
        opt.add_option('ls_maxiter', 10,
                       desc='Maximum number of line searches.')
        opt.add_option('alpha', 1.0,
                       desc='Initial over-relaxation factor.')


    def solve(self, params, unknowns, resids, system):
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
        """

        mode = system.ln_solver.options['mode']
        atol = self.options['atol']
        rtol = self.options['rtol']
        maxiter = self.options['maxiter']
        ls_atol = self.options['ls_atol']
        ls_rtol = self.options['ls_rtol']
        ls_maxiter = self.options['ls_maxiter']
        alpha = self.options['alpha']

        # perform an initial run
        system.apply_nonlinear(params, unknowns, resids)

        f_norm = resids.norm()
        f_norm0 = f_norm
        print('Residual:', f_norm)

        itercount = 0
        alpha_base = alpha
        while itercount < maxiter and f_norm > atol and \
              f_norm/f_norm0 > rtol:

            # Linearize Model
            system.jacobian(params, unknowns, resids)

            # Calculate direction to take step
            if mode == 'fwd':
                system.drmat[None].vec[:] = -resids.vec[:]
            else:
                system.dumat[None].vec[:] = -resids.vec[:]

            system.solve_linear(system.dumat, system.drmat, [None], mode=mode)
            dresids = system.drmat[None]

            #print "LS 1", uvec.array, '+', dfvec.array
            unknowns.vec[:] += alpha*dresids.vec[:]

            # Just evaluate the model with the new points
            system.apply_nonlinear(params, unknowns, resids)

            f_norm = resids.norm()
            print('Residual:', f_norm)

            itercount += 1
            ls_itercount = 0

            # Backtracking Line Search
            while ls_itercount < ls_maxiter and \
                  f_norm > ls_atol and \
                  f_norm/f_norm0 > ls_rtol:

                alpha *= 0.5
                unknowns.vec[:] -= alpha*dresids.vec[:]

                # Just evaluate the model with the new points
                system.apply_nonlinear(params, unknowns, resids)

                f_norm = resids.norm()

                ls_itercount += 1

            # Reset backtracking
            alpha = alpha_base

        # Need to make sure the whole workflow is executed at the final
        # point, not just evaluated.
        system.children_solve_nonlinear()
""" OpenMDAO LinearSolver that uses Scipy's GMRES to solve for derivatives."""

from __future__ import print_function

from six import iteritems

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from openmdao.solvers.solver_base import LinearSolver


class ScipyGMRES(LinearSolver):
    """ Scipy's GMRES Solver. This is a serial solver, so
    it should never be used in an MPI setting.
    """

    def __init__(self):
        super(ScipyGMRES, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12,
                       desc='Absolute convergence tolerance.')
        opt.add_option('maxiter', 100,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +

                       "let OpenMDAO determine the best mode.")
        opt.add_option('precondition', False,
                       desc='Set to True to turn on preconditioning.')

        # These are defined whenever we call solve to provide info we need in
        # the callback.
        self.system = None
        self.voi = None
        self.mode = None

    def solve(self, rhs_mat, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Args
        ----
        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        dict of ndarray : Solution vectors
        """

        options = self.options
        self.mode = mode

        unknowns_mat = {}
        for voi, rhs in iteritems(rhs_mat):

            # Scipy can only handle one right-hand-side at a time.
            self.voi = voi

            n_edge = len(rhs)
            A = LinearOperator((n_edge, n_edge),
                               matvec=self.mult,
                               dtype=float)

            # Support a preconditioner
            if self.options['precondition'] == True:
                M = LinearOperator((n_edge, n_edge),
                                   matvec=self.precon,
                                   dtype=float)
            else:
                M = None

            # Call GMRES to solve the linear system
            self.system = system
            d_unknowns, info = gmres(A, rhs, M=M,
                                     tol=options['atol'],
                                     maxiter=options['maxiter'])
            self.system = None

            if info > 0:
                msg = "ERROR in solve in '{}': gmres failed to converge " \
                      "after {} iterations"
                print(msg.format(system.name, options['maxiter']))
                #logger.error(msg, system.name, info)
            elif info < 0:
                msg = "ERROR in solve in '{}': gmres failed"
                print(msg.format(system.name))
                #logger.error(msg, system.name)

            unknowns_mat[voi] = d_unknowns

            #print(system.name, 'Linear solution vec', d_unknowns)


        return unknowns_mat

    def mult(self, arg):
        """ GMRES Callback: applies Jacobian matrix. Mode is determined by the
        system.

        Args
        ----
        arg : ndarray
            Incoming vector

        Returns
        -------
        ndarray : Matrix vector product of arg with jacobian
        """

        system = self.system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            sol_vec, rhs_vec = system.dumat[voi], system.drmat[voi]
        else:
            sol_vec, rhs_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        sol_vec.vec[:] = arg[:]

        # Start with a clean slate
        rhs_vec.vec[:] = 0.0
        system.clear_dparams()

        system.apply_linear(mode, ls_inputs=self.system._ls_inputs, vois=[voi])

        #print("arg", arg)
        #print("result", rhs_vec.vec)

        return rhs_vec.vec[:]

    def precon(self, arg):
        """ GMRES Callback: applies a preconditioner by calling
        solve_nonlinear on this system's children.

        Args
        ----
        arg : ndarray
            Incoming vector

        Returns
        -------
        ndarray : Preconditioned vector
        """

        system = self.system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            sol_vec, rhs_vec = system.dumat[voi], system.drmat[voi]
        else:
            sol_vec, rhs_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        rhs_vec.vec[:] = arg[:]

        # Start with a clean slate
        system.clear_dparams()

        for sub in system.subsystems():

            dumat = {}
            dumat[voi] = sub.dumat[voi]
            drmat = {}
            drmat[voi] = sub.drmat[voi]

            sub.solve_linear(dumat, drmat, (voi, ), mode=mode)

        #print("arg", arg)
        #print("preconditioned arg", precon_rhs)
        return sol_vec.vec

""" OpenMDAO LinearSolver that uses Scipy's GMRES to solve for derivatives."""

from __future__ import print_function

# pylint: disable=E0611, F0401
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from openmdao.solvers.solverbase import LinearSolver
from openmdao.devtools.debug import debug


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
        opt.add_option('mode', 'fwd', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " + \
                       "forward mode, 'rev' for reverse mode, or 'auto' to " + \
                       "let OpenMDAO determine the best mode.")

    def solve(self, rhs, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Parameters
        ----------
        rhs : ndarray
            Array containing the right-hand side for the linear solve. Also
            possibly a 2D array with multiple right-hand sides.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        ndarray : Solution vector
        """

        #TODO: When to record?

        n_edge = len(rhs)
        A = LinearOperator((n_edge, n_edge),
                           matvec=self.mult,
                           dtype=float)

        self.system = system
        options = self.options
        self.mode = mode

        # Call GMRES to solve the linear system
        d_unknowns, info = gmres(A, rhs,
                                 tol=options['atol'],
                                 maxiter=options['maxiter'])

        # TODO: Talk about warn/error logging
        if info > 0:
            msg = "ERROR in solve in '%s': gmres failed to converge " \
                  "after %d iterations"
            print(msg)
            #logger.error(msg, system.name, info)
        elif info < 0:
            msg = "ERROR in solve in '%s': gmres failed"
            print(msg)
            #logger.error(msg, system.name)

        #print system.name, 'Linear solution vec', d_unknowns
        self.system = None
        return d_unknowns

    def mult(self, arg):
        """ GMRES Callback: applies Jacobian matrix. Mode is determined by the
        system."""

        system = self.system
        mode = self.mode

        # FIXME: dumat/drmat keys won't always be None
        if mode=='fwd':
            sol_vec, rhs_vec = system.dumat[None], system.drmat[None]
        else:
            sol_vec, rhs_vec = system.drmat[None], system.dumat[None]

        # Set incoming vector
        sol_vec.vec[:] = arg[:]

        # Start with a clean slate
        rhs_vec.vec[:] = 0.0
        system.clear_dparams()

        system.apply_linear(system.params, system.unknowns, system.dpmat[None],
                            system.dumat[None], system.drmat[None], mode)

        #debug("arg", arg)
        #debug("result", rhs_vec.vec)
        return rhs_vec.vec[:]

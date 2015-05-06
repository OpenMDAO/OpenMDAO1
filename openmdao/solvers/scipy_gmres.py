""" OpenMDAO LinearSolver that uses Scipy's GMRES to solve for derivatives."""


# pylint: disable=E0611, F0401
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from openmdao.solvers.solverbase import LinearSolver


class ScipyGMRES(LinearSolver):
    """ Scipy's GMRES Solver. This is a serial solver, so
    it should never be used in an MPI setting.
    """

    def solve(self, rhs, system):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        rhs: ndarray
            Array containing the right hand side for the linear solve. Also
            possibly a 2D array with multiple right hand sides.
        """

        A = LinearOperator((n_edge, n_edge),
                           matvec=self.mult,
                           dtype=float)

        # TODO: Options dictionary?
        self.system = system
        options = self.options

        # Call GMRES to solve the linear system
        d_unknowns, info = gmres(A, rhs,
                                 tol=options['atol'],
                                 maxiter=options['maxiter'])

        # TODO: Talk about warn/error logging
        if info > 0:
            msg = "ERROR in solve in '%s': gmres failed to converge " \
                  "after %d iterations"
            #logger.error(msg, system.name, info)
        elif info < 0:
            msg = "ERROR in solve in '%s': gmres failed"
            #logger.error(msg, system.name)

        #print system.name, 'Linear solution vec', d_unknowns
        self.system = None
        return d_unknowns

    def mult(self, arg):
        """ GMRES Callback: applies Jacobian matrix. Mode is determined by the
        system."""

        # Set incoming vector
        system = self.system
        system.sol_vec.array[:] = arg[:]

        # Start with a clean slate
        system.rhs_vec.array[:] = 0.0
        system.varmanager.dparams[:] = 0.0

        # TODO: Rename this?
        system.applyJ()

        # TODO: Rename rhs_vec and sol_vec?
        return system.rhs_vec.array[:]

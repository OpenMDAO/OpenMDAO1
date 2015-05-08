""" OpenMDAO LinearSolver that uses Scipy's GMRES to solve for derivatives."""


# pylint: disable=E0611, F0401
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from openmdao.solvers.solverbase import LinearSolver


class ScipyGMRES(LinearSolver):
    """ Scipy's GMRES Solver. This is a serial solver, so
    it should never be used in an MPI setting.
    """

    def solve(self, rhs, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Parameters
        ----------
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
        self.mode = mode

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

        system = self.system
        mode = self.mode

        varmanager = system.varmanager
        params = varmanager.params
        unknowns = varmanager.unknowns
        resids = varmanager.resids
        dparams = varmanager.dparams
        dunknowns = varmanager.dunknowns
        dresids = varmanager.dresids

        if mode=='fwd':
            sol_vec, rhs_vec = unknowns, resids
        else:
            sol_vec, rhs_vec = resids, unknowns

        # Set incoming vector
        sol_vec.vec[:] = arg[:]

        # Start with a clean slate
        rhs_vec.vec[:] = 0.0
        dparams.vec[:] = 0.0

        system.apply_linear(params, unknowns, dparams, dunknowns, dresids,
                            mode)

        # TODO: Rename rhs_vec and sol_vec?
        return rhs_vec.vec[:]

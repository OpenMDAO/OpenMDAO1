""" OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
derivatives. This solver can be used under MPI."""

from __future__ import print_function

# TODO: Do we have to make this solver with a factory?
import petsc4py
from petsc4py import PETSc
import numpy

from openmdao.solvers.solver_base import LinearSolver


# This class object is given to KSP as a callback object for printing the residual.
class Monitor(object):
    """ Prints output from PETSc's KSP solvers """

    def __init__(self, ksp):
        """ Stores pointer to the ksp solver """
        self._ksp = ksp
        self._norm0 = 1.0

    def __call__(self, ksp, counter, norm):
        """ Store norm if first iteration, and print norm """
        if counter == 0 and norm != 0.0:
            self._norm0 = norm

        if self._ksp.options.iprint > 0:
            self._ksp.print_norm(self._ksp.ln_string, counter, norm, self._norm0)


class PetscKSP(LinearSolver):
    """ OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
    derivatives. This solver can be used under MPI."""

    def __init__(self):
        super(ScipyGMRES, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12,
                       desc='Absolute convergence tolerance.')
        opt.add_option('atol', 1e-12,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 100,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'fwd', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " + \
                       "forward mode, 'rev' for reverse mode, or 'auto' to " + \
                       "let OpenMDAO determine the best mode.")

        # These are defined whenever we call solve to provide info we need in
        # the callback.
        self.system = None
        self.voi = None
        self.mode = None

        self.ksp = None

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

        unknowns_mat = {}
        for voi, rhs in iteritems(rhs_mat):

            # Petsc can only handle one right-hand-side at a time for now
            self.voi = voi

            unknowns_mat[voi] = d_unknowns

            #print system.name, 'Linear solution vec', d_unknowns

        self.system = None
        return unknowns_mat

    def mult(self, mat, arg, result):
        """ KSP Callback: applies Jacobian matrix. Mode is determined by the
        system."""

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

        result[:] = rhs_vec.vec[:]


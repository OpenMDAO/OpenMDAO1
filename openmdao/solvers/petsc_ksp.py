""" OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
derivatives. This solver can be used under MPI."""

from __future__ import print_function
from six import iteritems, iterkeys

import os

# TODO: Do we have to make this solver with a factory?
import petsc4py
from petsc4py import PETSc
import numpy as np
from collections import OrderedDict

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import LinearSolver

trace = os.environ.get("OPENMDAO_TRACE")
if trace:  # pragma: no cover
    from openmdao.core.mpi_wrap import debug


def _get_petsc_vec_array_new(vec):
    """ Helper function to handle a petsc backwards incompatibility between 3.6
    and older versions."""

    return vec.getArray(readonly=True)


def _get_petsc_vec_array_old(vec):
    """ Helper function to handle a petsc backwards incompatibility between 3.6
    and older versions."""

    return vec.getArray()

try:
    petsc_version = petsc4py.__version__
except AttributeError:  # hack to fix doc-tests
    petsc_version = "3.5"

if int((petsc_version).split('.')[1]) >= 6:
    _get_petsc_vec_array = _get_petsc_vec_array_new
else:
    _get_petsc_vec_array = _get_petsc_vec_array_old


# This class object is given to KSP as a callback object for printing the residual.
class Monitor(object):
    """ Prints output from PETSc's KSP solvers """

    def __init__(self, ksp):
        """ Stores pointer to the ksp solver """
        self._ksp = ksp
        self._norm0 = 1.0
        self._norm = 1.0

    def __call__(self, ksp, counter, norm):
        """ Store norm if first iteration, and print norm """
        if counter == 0 and norm != 0.0:
            self._norm0 = norm
        self._norm = norm

        ksp = self._ksp
        ksp.iter_count += 1

        if ksp.options['iprint'] == 2:
            ksp.print_norm(ksp.print_name, ksp.system.pathname, ksp.iter_count,
                           norm, self._norm0, indent=1, solver='LN')


class PetscKSP(LinearSolver):
    """ OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
    derivatives. This solver can be used under MPI.

    Options
    -------
    options['atol'] :  float(1e-12)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print iteration totals to
        stdout, set to 2 to print the residual each iteration to stdout.
    options['maxiter'] :  int(100)
        Maximum number of iterations.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['rtol'] :  float(1e-12)
        Relative convergence tolerance.

    """

    def __init__(self):
        super(PetscKSP, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-12, lower=0.0,
                       desc='Relative convergence tolerance.')
        opt.add_option('maxiter', 100, lower=0,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.",
                       lock_on_setup=True)

        # These are defined whenever we call solve to provide info we need in
        # the callback.
        self.system = None
        self.voi = None
        self.mode = None

        self.ksp = {}
        self.print_name = 'KSP'

        # User can specify another linear solver to use as a preconditioner
        self.preconditioner = None

    def setup(self, system):
        """ Setup petsc problem just once.

        Args
        ----
        system : `System`
            Parent `System` object.
        """

        if not system.is_active():
            return

        # allocate and cache the ksp problem for each voi
        for voi in iterkeys(system.dumat):
            sizes = system._local_unknown_sizes[voi]
            lsize = np.sum(sizes[system.comm.rank, :])
            size = np.sum(sizes)

            if trace: debug("creating petsc matrix of size (%d,%d)" % (lsize, size))
            jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                               comm=system.comm)
            if trace: debug("petsc matrix creation DONE for %s" % voi)
            jac_mat.setPythonContext(self)
            jac_mat.setUp()

            if trace:  # pragma: no cover
                debug("creating KSP object for system", system.pathname)

            ksp = self.ksp[voi] = PETSc.KSP().create(comm=system.comm)
            if trace: debug("KSP creation DONE")

            ksp.setOperators(jac_mat)
            ksp.setType('fgmres')
            ksp.setGMRESRestart(1000)
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
            ksp.setMonitor(Monitor(self))

            if trace:  # pragma: no cover
                debug("ksp.getPC()")
                debug("rhs_buf, sol_buf size: %d" % lsize)
            pc_mat = ksp.getPC()
            pc_mat.setType('python')
            pc_mat.setPythonContext(self)

        if trace:  # pragma: no cover
            debug("ksp setup done")

        if self.preconditioner:
            self.preconditioner.setup(system)

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

        unknowns_mat = OrderedDict()
        maxiter = options['maxiter']
        atol = options['atol']
        rtol = options['rtol']

        for voi, rhs in iteritems(rhs_mat):

            ksp = self.ksp[voi]

            ksp.setTolerances(max_it=maxiter, atol=atol, rtol=rtol)

            sol_vec = np.zeros(rhs.shape)
            # Set these in the system
            if trace:  # pragma: no cover
                debug("creating sol_buf petsc vec for voi", voi)
            self.sol_buf_petsc = PETSc.Vec().createWithArray(sol_vec,
                                                             comm=system.comm)
            if trace:  # pragma: no cover
                debug("sol_buf creation DONE")
                debug("creating rhs_buf petsc vec for voi", voi)
            self.rhs_buf_petsc = PETSc.Vec().createWithArray(rhs,
                                                             comm=system.comm)
            if trace: debug("rhs_buf creation DONE")

            # Petsc can only handle one right-hand-side at a time for now
            self.voi = voi
            self.system = system
            self.iter_count = 0
            ksp.solve(self.rhs_buf_petsc, self.sol_buf_petsc)
            self.system = None

            # Final residual print if you only want the last one
            if self.options['iprint'] == 1:
                mon = ksp.getMonitor()[0][0]
                self.print_norm(self.print_name, system.pathname, self.iter_count,
                                mon._norm, mon._norm0, indent=1, solver='LN')

            if self.iter_count >= maxiter:
                msg = 'FAILED to converge in %d iterations' % self.iter_count
                fail = True
            else:
                msg = 'Converged in %d iterations' % self.iter_count
                fail = False

            if fail or self.options['iprint'] > 0:
                self.print_norm(self.print_name, system.pathname,
                                self.iter_count, 0, 0, msg=msg, indent=1, solver='LN')

            unknowns_mat[voi] = sol_vec

            if fail and self.options['err_on_maxiter']:
                raise AnalysisError("Solve in '%s': PetscKSP %s" %
                                    (system.pathname, msg))

            #print system.name, 'Linear solution vec', d_unknowns

        self.system = None
        return unknowns_mat

    def mult(self, mat, arg, result):
        """ KSP Callback: applies Jacobian matrix. Mode is determined by the
        system.

        Args
        ----
        arg : PetSC Vector
            Incoming vector

        result : PetSC Vector
            Empty array into which we place the matrix-vector product.
        """

        system = self.system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            sol_vec, rhs_vec = system.dumat[voi], system.drmat[voi]
        else:
            sol_vec, rhs_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        # sol_vec.vec[:] = arg.array
        sol_vec.vec[:] = _get_petsc_vec_array(arg)

        # Start with a clean slate
        rhs_vec.vec[:] = 0.0
        system.clear_dparams()

        system._sys_apply_linear(mode, self.system._do_apply, vois=(voi,))

        result.array[:] = rhs_vec.vec

        # print("arg", arg.array)
        # print("result", result.array)

    def apply(self, mat, arg, result):
        """ Applies preconditioner.

        Args
        ----
        arg : PetSC Vector
            Incoming vector

        result : PetSC Vector
            Empty vector into which we return the preconditioned arg
        """

        if self.preconditioner is None:
            result.array[:] = _get_petsc_vec_array(arg)
            return

        system = self.system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            sol_vec, rhs_vec = system.dumat[voi], system.drmat[voi]
        else:
            sol_vec, rhs_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        rhs_vec.vec[:] = _get_petsc_vec_array(arg)

        # Start with a clean slate
        system.clear_dparams()

        dumat = OrderedDict()
        dumat[voi] = system.dumat[voi]
        drmat = OrderedDict()
        drmat[voi] = system.drmat[voi]

        with system._dircontext:
            system.solve_linear(dumat, drmat, (voi, ), mode=mode,
                                solver=self.preconditioner)

        result.array[:] = sol_vec.vec

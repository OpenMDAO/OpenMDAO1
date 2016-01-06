""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve. Inherits from ScipyGMRES just for the mult function."""

import numpy as np

from openmdao.solvers.scipy_gmres import ScipyGMRES
from collections import OrderedDict


class DirectSolver(ScipyGMRES):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve.

    Options
    -------
    options['atol'] :  float(1e-12)
        Absolute convergence tolerance.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(1000)
        Maximum number of iterations.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['precondition'] :  bool(False)
        Set to True to turn on preconditioning.

    """

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
        sol_buf = OrderedDict()

        # TODO: This solver could probably work with multiple RHS
        for voi, rhs in rhs_mat.items():
            self.voi = None

            # TODO: When to record?
            self.system = system
            self.mode = mode

            n_edge = len(rhs)
            ident = np.eye(n_edge)

            partials = np.empty((n_edge, n_edge))

            for i in range(n_edge):
                partials[:, i] = self.mult(ident[:, i])

            deriv = np.linalg.solve(partials, rhs)

            self.system = None
            sol_buf[voi] = deriv

        return sol_buf

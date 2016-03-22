""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve. Inherits from ScipyGMRES just for the mult function."""

import numpy as np

from openmdao.solvers.solver_base import MultLinearSolver
from collections import OrderedDict


class DirectSolver(MultLinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve.

    """

    def __init__(self):
        super(DirectSolver, self).__init__()
        self.options.remove_option("err_on_maxiter")
        self.options.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.")

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

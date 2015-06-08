""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve."""

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.solvers.solverbase import LinearSolver


class ExplicitSolver(LinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve."""

    def solve(self, rhs, system, mode):

        n_edge = len(rhs)
        I = np.eye(n_edge)

        partials = np.empty((n_edge, n_edge))

        for i in range(n_edge):
            partials[:, i] = self.mult(I[:, i])

        deriv = np.linalg.solve(partials, rhs)

        return deriv

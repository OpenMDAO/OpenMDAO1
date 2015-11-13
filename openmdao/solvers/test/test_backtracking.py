""" Unit testing for the Backtracking sub-solver. """

import unittest

from openmdao.api import Problem, Group, NonLinearSolver, IndepVarComp
from openmdao.solvers.backtracking import BackTracking
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.util import assert_rel_error


class FakeSolver(NonLinearSolver):
    """ Does nothing but invoke a line search."""

    def __init__(self):
        super(FakeSolver, self).__init__()
        self.sub = BackTracking()

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Calc deriv then do line search."""

        # Perform an initial run to propagate srcs to targets.
        system.children_solve_nonlinear(None)
        system.apply_nonlinear(params, unknowns, resids)

        # Linearize Model with partial derivatives
        system._sys_linearize(params, unknowns, resids, total_derivs=False)

        # Calculate direction to take step
        arg = system.drmat[None]
        result = system.dumat[None]

        # Step waaaaay to far so we have to backtrack
        arg.vec[:] = resids.vec*100
        system.solve_linear(system.dumat, system.drmat, [None], mode='fwd')

        unknowns.vec += result.vec

        self.sub.solve(params, unknowns, resids, system, self, 1.0, 1.0, 1.0)

class TestBackTracking(unittest.TestCase):

    def test_newton_with_backtracking(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.root.nl_solver.line_search.options['atol'] = 1e-12
        top.root.nl_solver.line_search.options['rtol'] = 1e-12
        top.root.nl_solver.line_search.options['maxiter'] = 3

        # This is a very contrived test, but we step 8 times farther than we
        # should, then allow the line search to backtrack 3 steps, which
        # takes us back to 1.0.
        top.root.nl_solver.options['alpha'] = 8.0

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['state_eq.y2_command'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()
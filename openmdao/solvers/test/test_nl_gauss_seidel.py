""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

from openmdao.core.problem import Problem
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.test.sellar import SellarNoDerivatives
from openmdao.test.testutil import assert_rel_error

class TestNLGaussSeidel(unittest.TestCase):

    def test_sellar(self):

        top = Problem()
        top.root = SellarNoDerivatives()
        top.root.nl_solver = NLGaussSeidel()

        top.setup()
        top.run()

        unknowns = top.root._varmanager.unknowns
        y1 = unknowns['y1']
        y2 = unknowns['y2']

        assert_rel_error(self, y1, 25.58830273, .00001)
        assert_rel_error(self, y2, 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)

if __name__ == "__main__":
    unittest.main()

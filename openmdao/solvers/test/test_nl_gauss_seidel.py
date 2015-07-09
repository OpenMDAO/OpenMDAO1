""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

from openmdao.core.problem import Problem
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.test.sellar import SellarNoDerivatives, SellarDerivativesGrouped
from openmdao.test.testutil import assert_rel_error

class TestNLGaussSeidel(unittest.TestCase):

    def test_sellar(self):

        top = Problem()
        top.root = SellarNoDerivatives()
        top.root.nl_solver = NLGaussSeidel()

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)

    def test_sellar_group(self):

        top = Problem()
        top.root = SellarDerivativesGrouped()
        top.root.nl_solver = NLGaussSeidel()
        top.root.nl_solver.options['atol'] = 1e-9
        top.root.mda.nl_solver.options['atol'] = 1e-3

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()

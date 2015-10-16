""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

from openmdao.api import Problem, NLGaussSeidel
from openmdao.test.sellar import SellarNoDerivatives, SellarDerivativesGrouped
from openmdao.test.util import assert_rel_error


class TestNLGaussSeidel(unittest.TestCase):

    def test_sellar(self):

        prob = Problem()
        prob.root = SellarNoDerivatives()
        prob.root.nl_solver = NLGaussSeidel()

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.root.nl_solver.iter_count, 8)

    def test_sellar_group(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()
        prob.root.nl_solver = NLGaussSeidel()
        prob.root.nl_solver.options['atol'] = 1e-9
        prob.root.mda.nl_solver.options['atol'] = 1e-3
        prob.root.nl_solver.options['iprint'] = 1 # so that print_norm is in coverage

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()

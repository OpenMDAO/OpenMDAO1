""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import sys
import unittest

from six.moves import cStringIO

from openmdao.api import Problem, NLGaussSeidel, AnalysisError
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

        # Make sure we only call apply_linear on 'heads'
        nd1 = prob.root.cycle.d1.execution_count
        nd2 = prob.root.cycle.d2.execution_count
        if prob.root.cycle.d1._run_apply == True:
            self.assertEqual(nd1, 2*nd2)
        else:
            self.assertEqual(2*nd1, nd2)

    def test_sellar_analysis_error(self):

        prob = Problem()
        prob.root = SellarNoDerivatives()
        prob.root.nl_solver = NLGaussSeidel()
        prob.root.nl_solver.options['maxiter'] = 2
        prob.root.nl_solver.options['err_on_maxiter'] = True

        prob.setup(check=False)

        try:
            prob.run()
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': NLGaussSeidel FAILED to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_sellar_group(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()
        prob.root.nl_solver = NLGaussSeidel()
        prob.root.nl_solver.options['atol'] = 1e-9
        prob.root.mda.nl_solver.options['atol'] = 1e-3
        prob.root.nl_solver.options['iprint'] = 1 # so that print_norm is in coverage

        prob.setup(check=False)

        old_stdout = sys.stdout
        sys.stdout = cStringIO() # so we don't see the iprint output during testing
        try:
            prob.run()
        finally:
            sys.stdout = old_stdout

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()

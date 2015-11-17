"""
Who is Testing the Testers?
"""

import logging
import os.path
import sys

import unittest
import numpy as np

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.test.util import assert_rel_error, problem_derivatives_check
from openmdao.test.simple_comps import SimpleCompWrongDeriv


class TestCase(unittest.TestCase):
    """ Test Test functions. """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_rel_error_inf_nan(self):

        try:
            assert_rel_error(self, float('nan'), 6.5, 0.0001)
        except AssertionError as exc:
            msg = "actual nan, desired 6.5, rel error nan, tolerance 0.0001"
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected AssertionError')

        try:
            assert_rel_error(self, float('inf'), 6.5, 0.0001)
        except AssertionError as exc:
            msg = "actual inf, desired 6.5, rel error inf, tolerance 0.0001"
            self.assertEqual(str(exc), msg)
        else:
            self.fail('Expected AssertionError')

        # We may actually want this to work for some tests.
        assert_rel_error(self, float('nan'), float('nan'), 0.0001)

    def test_rel_error_array(self):

        try:
            assert_rel_error(self, 1e-2*np.ones(3), np.zeros(3), 1e-3)
        except AssertionError as exc:
            msg="arrays do not match, rel error 1.732e-02 > tol (1.000e-03)"
            self.assertEqual(msg, str(exc))
        else:
            self.fail("Expected Assertion Error")

        err = assert_rel_error(self, 1e-2*np.ones(3), 1e-2*np.ones(3), 1e-10)
        self.assertEqual(err, 0.0)

        err = assert_rel_error(self, 1e-2*np.ones(3), 1.00001e-2*np.ones(3), 1e-3)
        self.assertLessEqual(err, 1e-5)


    def test_problem_deriv_test(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleCompWrongDeriv())
        prob.root.add('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        # suppress printed output from problem_derivatives_check()
        sysout = sys.stdout
        devnull = open(os.devnull, 'w')

        try:
            sys.stdout = devnull
            problem_derivatives_check(self, prob)
        except AssertionError as err:
            sys.stdout = sysout
            self.assertIn("not less than or equal to 1e-05", err.args[0])
        finally:
            sys.stdout = sysout



if __name__ == "__main__":
    unittest.main()

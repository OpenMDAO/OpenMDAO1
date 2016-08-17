""" Test out our new error handler helper. """

import sys
import unittest

from six.moves import cStringIO
import numpy as np

from openmdao.api import Problem, Group, Component, NLGaussSeidel, ScipyGMRES


class ErrorComp(Component):
    """ This component generates numpy errors."""

    def __init__(self, mode):
        super(ErrorComp, self).__init__()

        self.add_param('in', val=np.array([1.0]))
        self.add_output('out', val=np.array([1.0]))

        self.mode = mode

        self.deriv_options['type'] = 'fd'

    def solve_nonlinear(self, params, unknowns, resids):
        """Produce various types of errors."""

        if self.mode == 'divide':
            unknowns['out'] = params['in']/0.0
        if self.mode == 'indirect_divide':
            unknowns['out'] = np.array([np.inf])
        else:
            unknowns['out'] = params['in']/0.99


class TestNPError(unittest.TestCase):

    def tearDown(self):
        # put numpy error behavior back to default
        np.seterr(over='warn')
        np.seterr(divide='warn')
        np.seterr(invalid='warn')

    def test_direct_errors_divide(self):

        top = Problem(debug=True)
        top.root = root = Group()
        root.add('comp', ErrorComp('divide'))

        top.setup(check=False)

        with self.assertRaises(FloatingPointError) as err:
            top.run()

        expected_msg = "divide by zero encountered in divide"
        msg = str(err.exception)
        msg = msg.replace('in true_divide', 'in divide')
        self.assertEqual(msg, expected_msg)

    def test_indirect_errors_divide(self):

        top = Problem(debug=True)
        top.root = root = Group()
        root.add('comp1', ErrorComp('xx'))
        root.add('comp2', ErrorComp('indirect_divide'))
        root.connect('comp1.out', 'comp2.in')
        root.connect('comp2.out', 'comp1.in')

        root.nl_solver = NLGaussSeidel()
        root.ln_solver = ScipyGMRES()

        top.setup(check=False)

        with self.assertRaises(FloatingPointError) as err:
            top.run()

        expected_msg = "invalid value encountered in subtract\nThe following unknowns are nonfinite: ['comp1.out', 'comp2.out']\nThe following resids are nonfinite: ['comp1.out']\nThe following params are nonfinite: ['comp1.in']"
        self.assertEqual(str(err.exception), expected_msg)

    def test_indirect_errors_divide_subbed(self):

        # Make sure that two stacked solvers don't double append.

        top = Problem(debug=True)
        top.root = root = Group()
        sub = root.add('sub', Group())
        sub.add('comp1', ErrorComp('xx'))
        sub.add('comp2', ErrorComp('indirect_divide'))
        root.connect('sub.comp1.out', 'sub.comp2.in')
        root.connect('sub.comp2.out', 'sub.comp1.in')

        root.nl_solver = NLGaussSeidel()
        root.ln_solver = ScipyGMRES()
        sub.nl_solver = NLGaussSeidel()
        sub.ln_solver = ScipyGMRES()

        top.setup(check=False)

        with self.assertRaises(FloatingPointError) as err:
            top.run()

        expected_msg = "invalid value encountered in subtract\nThe following unknowns are nonfinite: ['comp1.out', 'comp2.out']\nThe following resids are nonfinite: ['comp1.out']\nThe following params are nonfinite: ['comp1.in']"
        self.assertEqual(str(err.exception), expected_msg)


if __name__ == "__main__":
    unittest.main()

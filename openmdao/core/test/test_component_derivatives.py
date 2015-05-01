""" Test for the Component class. This test just contains derivatives related
tests. Note: these are isolated/harnessed tests, so they won't involve any of
the derivatives system outside of Component."""

import unittest

import numpy as np

from openmdao.core.component import Component
from openmdao.test.simplecomps import SimpleCompDerivJac, SimpleArrayComp


class TestComponentDerivatives(unittest.TestCase):

    def test_simple_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.

        empty = {}
        mycomp = SimpleCompDerivJac()
        mycomp.linearize(empty, empty)

        # Forward

        dparams = {}
        dparams['x'] = np.array([3.1])
        dunknowns = {}
        dunknowns['y'] = np.array([0.0])

        mycomp.apply_linear(empty, empty, empty, dparams, dunknowns,
                            empty, 'fwd')

        self.assertEqual(dunknowns['y'], 6.2)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0])
        dunknowns = {}
        dunknowns['y'] = np.array([3.1])

        mycomp.apply_linear(empty, empty, empty, dparams, dunknowns,
                            empty, 'rev')

        self.assertEqual(dparams['x'], 6.2)

    def test_simple_array_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.
        # Now with arrays

        empty = {}
        mycomp = SimpleArrayComp()
        mycomp.linearize(empty, empty)

        # Forward

        dparams = {}
        dparams['x'] = np.array([1.5, 3.1])
        dunknowns = {}
        dunknowns['y'] = np.array([0.0, 0.0])

        mycomp.apply_linear(empty, empty, empty, dparams, dunknowns,
                            empty, 'fwd')
        target = mycomp.J[('y', 'x')].dot(dparams['x'])
        diff = abs(dunknowns['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0, 0.0])
        dunknowns = {}
        dunknowns['y'] = np.array([1.5, 3.1])

        mycomp.apply_linear(empty, empty, empty, dparams, dunknowns,
                            empty, 'rev')
        target = mycomp.J[('y', 'x')].T.dot(dunknowns['y'])
        diff = abs(dparams['x'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

if __name__ == "__main__":
    unittest.main()

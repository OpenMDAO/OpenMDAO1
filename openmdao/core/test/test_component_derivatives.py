""" Test for the Component class. This test just contains derivatives related
tests. Note: these are isolated/harnessed tests, so they won't involve any of
the derivatives system outside of Component."""

import unittest

import numpy as np

from openmdao.core.component import Component
from openmdao.test.simplecomps import SimpleCompDerivJac, SimpleArrayComp, \
                                      SimpleImplicitComp


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
        target = mycomp._jacobian_cache[('y', 'x')].dot(dparams['x'])
        diff = abs(dunknowns['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0, 0.0])
        dunknowns = {}
        dunknowns['y'] = np.array([1.5, 3.1])

        mycomp.apply_linear(empty, empty, empty, dparams, dunknowns,
                            empty, 'rev')
        target = mycomp._jacobian_cache[('y', 'x')].T.dot(dunknowns['y'])
        diff = abs(dparams['x'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

    def test_simple_implicit_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.
        # Now with a comp that has a state.

        params = {}
        params['x'] = 0.5
        unknowns = {}
        unknowns['y'] = 0.0
        unknowns['z'] = 0.0
        resids = {}
        resids['z'] = 0.0

        mycomp = SimpleImplicitComp()

        # Run model so we can calc derivatives around the solved state
        mycomp.solve_nonlinear(params, unknowns, resids)

        mycomp.linearize(params, unknowns)
        J = mycomp._jacobian_cache

        # Forward

        dparams = {}
        dparams['x'] = np.array([1.3])
        dstates = {}
        dstates['z'] = np.array([2.5])
        dunknowns = {}
        dunknowns['y'] = np.array([0.0])
        dunknowns['z'] = np.array([0.0])

        mycomp.apply_linear(params, unknowns, resids, dparams, dunknowns,
                            dstates, 'fwd')

        target = J[('y', 'x')]*dparams['x'] + J[('y', 'z')]*dstates['z']
        diff = abs(dunknowns['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        target = J[('z', 'x')]*dparams['x'] + J[('z', 'z')]*dstates['z']
        diff = abs(dunknowns['z'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0])
        dstates = {}
        dstates['z'] = np.array([0.0])
        dunknowns = {}
        dunknowns['y'] = np.array([1.5])
        dunknowns['z'] = np.array([2.3])

        mycomp.apply_linear(params, unknowns, resids, dparams, dunknowns,
                            dstates, 'rev')

        target = J[('y', 'x')]*dunknowns['y'] + J[('z', 'x')]*dunknowns['z']
        diff = abs(dparams['x'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        target = J[('y', 'z')]*dunknowns['y'] + J[('z', 'z')]*dunknowns['z']
        diff = abs(dstates['z'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

if __name__ == "__main__":
    unittest.main()

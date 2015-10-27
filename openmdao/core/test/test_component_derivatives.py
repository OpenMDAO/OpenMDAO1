""" Test for the Component class. This test just contains derivatives related
tests. Note: these are isolated/harnessed tests, so they won't involve any of
the derivatives system outside of Component."""

import unittest
import warnings

import numpy as np

from openmdao.api import Problem, Group, Component, ExecComp, IndepVarComp
from openmdao.test.simple_comps import SimpleComp, SimpleArrayComp, \
                                       SimpleImplicitComp, SimpleSparseArrayComp

from openmdao.test.util import assert_rel_error

class MyComp(SimpleComp):
    def jacobian(self, params, unknowns, resids):
        return {('y','x'): np.array([[2.0]])}

class TestComponentDerivatives(unittest.TestCase):

    def test_simple_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.

        empty = {}
        params = {'x': 0.0}
        unknowns = {'y': 0.0}
        mycomp = ExecComp(['y=2.0*x'])
        mycomp._jacobian_cache = mycomp.linearize(params, unknowns, empty)

        # Forward

        dparams = {}
        dparams['x'] = np.array([3.1])
        dresids = {}
        dresids['y'] = np.array([0.0])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'fwd')

        self.assertEqual(dresids['y'], 6.2)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0])
        dresids = {}
        dresids['y'] = np.array([3.1])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'rev')

        self.assertEqual(dparams['x'], 6.2)

    def test_simple_array_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.
        # Now with arrays

        empty = {}
        mycomp = SimpleArrayComp()
        mycomp._jacobian_cache = mycomp.linearize(empty, empty, empty)

        # Forward

        dparams = {}
        dparams['x'] = np.array([1.5, 3.1])
        dresids = {}
        dresids['y'] = np.array([0.0, 0.0])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'fwd')
        target = mycomp._jacobian_cache[('y', 'x')].dot(dparams['x'])
        diff = abs(dresids['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0, 0.0])
        dresids = {}
        dresids['y'] = np.array([1.5, 3.1])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'rev')
        target = mycomp._jacobian_cache[('y', 'x')].T.dot(dresids['y'])
        diff = abs(dparams['x'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

    def test_sparse_array_Jacobian(self):

        # Tests that we can correctly handle user-defined Jacobians.
        # Now with arrays

        empty = {}
        mycomp = SimpleSparseArrayComp()
        mycomp._jacobian_cache = mycomp.linearize(empty, empty, empty)

        # Forward

        dparams = {}
        dparams['x'] = np.array([1.5, 7.4, 3.1, 2.6])
        dresids = {}
        dresids['y'] = np.array([0.0, 0.0, 0.0, 0.0])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'fwd')
        target = mycomp._jacobian_cache[('y', 'x')].dot(dparams['x'])
        diff = abs(dresids['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0, 0.0, 0.0, 0.0])
        dresids = {}
        dresids['y'] = np.array([1.5, 7.4, 3.1, 2.6])

        mycomp.apply_linear(empty, empty, dparams, empty,
                            dresids, 'rev')
        target = mycomp._jacobian_cache[('y', 'x')].T.dot(dresids['y'])
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

        mycomp._jacobian_cache = mycomp.linearize(params, unknowns, resids)
        J = mycomp._jacobian_cache

        # Forward

        dparams = {}
        dparams['x'] = np.array([1.3])
        dunknowns = {}
        dunknowns['z'] = np.array([2.5])
        dresids = {}
        dresids['y'] = np.array([0.0])
        dresids['z'] = np.array([0.0])

        mycomp.apply_linear(params, unknowns, dparams, dunknowns,
                            dresids, 'fwd')

        target = J[('y', 'x')]*dparams['x'] + J[('y', 'z')]*dunknowns['z']
        diff = abs(dresids['y'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        target = J[('z', 'x')]*dparams['x'] + J[('z', 'z')]*dunknowns['z']
        diff = abs(dresids['z'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Reverse

        dparams = {}
        dparams['x'] = np.array([0.0])
        dunknowns = {}
        dunknowns['z'] = np.array([0.0])
        dresids = {}
        dresids['y'] = np.array([1.5])
        dresids['z'] = np.array([2.3])

        mycomp.apply_linear(params, unknowns, dparams, dunknowns,
                            dresids, 'rev')

        target = J[('y', 'x')]*dresids['y'] + J[('z', 'x')]*dresids['z']
        diff = abs(dparams['x'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

        target = J[('y', 'z')]*dresids['y'] + J[('z', 'z')]*dresids['z']
        diff = abs(dunknowns['z'] - target).max()
        self.assertAlmostEqual(diff, 0.0, places=3)

    def test_jacobian_deprecated(self):
        p = Problem(root=Group())
        p.root.add('P1' ,IndepVarComp('x', 1.0))
        p.root.add('comp', MyComp())
        p.root.connect('P1.x', 'comp.x')
        p.setup(check=False)

        indep_list = ['P1.x']
        unknown_list = ['comp.y']

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            J = p.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')

            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message),
                             "comp: The 'jacobian' method is deprecated. Please rename "
                             "'jacobian' to 'linearize'.")

        assert_rel_error(self, J['comp.y']['P1.x'][0][0], 2.0, 1e-6)
        self.assertEqual(J['comp.y']['P1.x'].size, 1)

if __name__ == "__main__":
    unittest.main()

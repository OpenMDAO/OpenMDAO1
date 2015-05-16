from __future__ import print_function

import unittest

import numpy as np

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem

from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleArrayComp, SimpleCompDerivJac, SimpleImplicitComp
from openmdao.test.testutil import assert_equal_jacobian


class TestProb(Problem):

    def __init__(self):
        super(TestProb, self).__init__()


        self.root = root = Group()
        root.add('c1', SimpleArrayComp())
        root.add('p1', ParamComp('p', 1*np.ones(2)))
        root.connect('p1:p','c1:x')

        root.add('ci1', SimpleImplicitComp())
        root.add('pi1', ParamComp('p', 1.))
        root.connect('pi1:p','ci1:x')

class SysFDTestCase(unittest.TestCase):

    def setUp(self):
        self.p = TestProb()
        self.p.setup()

    def test_correct_keys_in_jac(self):
        expected_keys=[('y','x'), ]
        params = {'x': np.ones(2)}
        unknowns = {'y': np.zeros(2)}
        resids = {'y': np.zeros(2)}
        jac = self.p['c1']._fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

    def test_correct_vals_in_jac(self):
        params = {'x': np.ones(2)}
        unknowns = {'y': np.zeros(2)}
        resids = {'y': np.zeros(2)}
        jac = self.p['c1']._fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2.,  7.],[ 5., -3.]])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

        #Got lucky that the way this comp was written, it would accept any square
        # matrix. But provided jacobian would be really wrong!
        params = {'x': np.ones((2,2))}
        unknowns = {'y': np.zeros((2,2))}
        resids = {'y': np.zeros((2,2))}
        jac = self.p['c1']._fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2,  0,  7,  0],
                                              [ 0,  2,  0,  7],
                                              [ 5,  0, -3,  0],
                                              [ 0,  5,  0, -3]
                                             ])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

    def test_correct_vals_in_jac_implicit(self):
        params = {'x': .5}
        unknowns = {'y': 0., 'z':0}
        resids = {'y':0., 'z': 0}

        jac = self.p['ci1']._fd_jacobian(params, unknowns, resids)
        expected_jac = {}
        # Output equation
        expected_jac[('y', 'x')] = 1.
        expected_jac[('y', 'z')] = 2.0
        # State equation
        expected_jac[('z', 'z')] = params['x'] + 1
        expected_jac[('z', 'x')] = unknowns['z']

        # print(jac)
        # print()
        # print(expected_jac)
        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

if __name__ == "__main__":
    unittest.main()

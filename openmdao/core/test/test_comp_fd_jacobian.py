""" Unit test for the fd_jacobian method on Component. This method peforms a
finite difference."""

from __future__ import print_function
from collections import OrderedDict
import unittest

import numpy as np

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.system import System
from openmdao.core.vecwrapper import SrcVecWrapper

from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleArrayComp, SimpleCompDerivJac, \
                                      SimpleImplicitComp, SimpleComp
from openmdao.test.testutil import assert_equal_jacobian, assert_rel_error


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


class CompFDTestCase(unittest.TestCase):
    """ Some basic tests of the fd_jacobian method in Component."""

    def setUp(self):
        self.p = TestProb()
        self.p.setup()

    def test_correct_keys_in_jac(self):

        expected_keys=[('y', 'x')]

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2)),
                             'pathname' : 'x',
                             'relative_name' : 'x' }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2)),
                               'pathname' : 'y',
                               'relative_name' : 'y' }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2)),
                             'pathname' : 'y',
                             'relative_name' : 'y' }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

    def test_correct_vals_in_jac(self):

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2)),
                             'pathname' : 'x',
                             'relative_name' : 'x' }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2)),
                               'pathname' : 'y',
                               'relative_name' : 'y' }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2)),
                             'pathname' : 'y',
                             'relative_name' : 'y' }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        self.p.subsystem('c1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2.,  7.],[ 5., -3.]])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

        #Got lucky that the way this comp was written, it would accept any square
        # matrix. But provided jacobian would be really wrong!

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2, 2)),
                             'pathname' : 'x',
                             'relative_name' : 'x' }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2, 2)),
                               'pathname' : 'y',
                               'relative_name' : 'y' }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2, 2)),
                             'pathname' : 'y',
                             'relative_name' : 'y' }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        self.p.subsystem('c1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2,  0,  7,  0],
                                              [ 0,  2,  0,  7],
                                              [ 5,  0, -3,  0],
                                              [ 0,  5,  0, -3]
                                             ])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

    def test_correct_vals_in_jac_implicit(self):

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.array([0.5]),
                             'pathname' : 'x',
                             'relative_name' : 'x' }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.array([0.0]),
                               'pathname' : 'y',
                               'relative_name' : 'y' }
        unknowns_dict['z'] = { 'val': np.array([0.0]),
                               'pathname' : 'z',
                               'relative_name' : 'z' }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.array([0.0]),
                             'pathname' : 'y',
                             'relative_name' : 'y' }
        resids_dict['z'] = { 'val': np.array([0.0]),
                             'pathname' : 'z',
                             'relative_name' : 'z' }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        self.p.subsystem('ci1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('ci1').fd_jacobian(params, unknowns, resids)
        expected_jac = {}
        # Output equation
        expected_jac[('y', 'x')] = 1.
        expected_jac[('y', 'z')] = 2.0
        # State equation
        expected_jac[('z', 'z')] = params['x'] + 1
        expected_jac[('z', 'x')] = unknowns['z']

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)


class CompFDinSystemTestCase(unittest.TestCase):
    """ Tests automatic finite difference of a component in a full problem."""

    def test_error_no_derivatives(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', SimpleComp())
        top.root.add('p1', ParamComp('x', 2.0))
        top.root.connect('p1:x', 'comp:x')

        comp.fd_options['force_fd'] = False

        top.setup()
        top.run()

        try:
            J = top.calc_gradient(['p1:x'], ['comp:y'], mode='fwd', return_format='dict')
        except Exception as err:
            msg = "No derivatives defined for Component 'comp'"
            self.assertEqual(str(err), msg)
        else:
            self.fail("Exception expected")

    def test_no_derivatives(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', SimpleComp())
        top.root.add('p1', ParamComp('x', 2.0))
        top.root.connect('p1:x', 'comp:x')

        comp.fd_options['force_fd'] = True

        top.setup()
        top.run()

        J = top.calc_gradient(['p1:x'], ['comp:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp:y']['p1:x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['p1:x'], ['comp:y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp:y']['p1:x'][0][0], 2.0, 1e-6)

if __name__ == "__main__":
    unittest.main()

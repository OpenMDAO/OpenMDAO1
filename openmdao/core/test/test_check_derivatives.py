""" Testing for Problem.check_partial_derivatives."""

from six import iteritems
import unittest

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.simplecomps import SimpleArrayComp, SimpleImplicitComp, \
                                      SimpleCompDerivMatVec
from openmdao.test.testutil import assert_rel_error


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()

        top.setup()
        top.run()

        data = top.check_partial_derivatives(out_stream=None)
        #print data

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleArrayComp())
        top.root.add('p1', ParamComp('x', np.ones([2])))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        data = top.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.add('p1', ParamComp('x', 0.5))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        data = top.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_bad_size(self):

        class BadComp(SimpleArrayComp):
            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives"""
                J = {}
                J[('y', 'x')] = np.zeros((3, 3))
                return J


        top = Problem()
        top.root = Group()
        top.root.add('comp', BadComp())
        top.root.add('p1', ParamComp('x', np.ones([2])))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        try:
            data = top.check_partial_derivatives(out_stream=None)
        except Exception as err:
            msg = "Jacobian in component 'comp' between the" + \
                " variables 'x' and 'y' is the wrong size. " + \
                "It should be 2 by 2"
            self.assertEquals(str(err), msg)
        else:
            self.fail("Error expected")


class TestProblemFullFD(unittest.TestCase):

    def test_full_model_fd_simple_comp(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleCompDerivMatVec())
        top.root.add('p1', ParamComp('x', 1.0))

        top.root.connect('p1:x', 'comp:x')

        top.root.fd_options['force_fd'] = True

        top.setup()
        top.run()

        param_list = ['comp:x']
        unknown_list = ['comp:y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp:y']['comp:x'][0][0], 2.0, 1e-6)


    def test_full_model_fd_simple_comp_promoted(self):

        top = Problem()
        top.root = Group()
        sub = top.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        top.root.add('p1', ParamComp('x', 1.0), promotes=['*'])

        top.root.fd_options['force_fd'] = True

        top.setup()
        top.run()

        param_list = ['x']
        unknown_list = ['y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        param_list = ['sub:comp:x']
        unknown_list = ['sub:comp:y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub:comp:y']['sub:comp:x'][0][0], 2.0, 1e-6)


    def test_full_model_fd_double_diamond_grouped(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()
        top.setup()
        top.run()

        top.root.fd_options['force_fd'] = True

        param_list = ['sub1:comp1:x1']
        unknown_list = ['comp7:y1']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['sub1:comp1:x1'][0][0], -40.75, 1e-6)

        print J
        top.root.fd_options['form'] = 'central'
        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        print J
        assert_rel_error(self, J['comp7:y1']['sub1:comp1:x1'][0][0], -40.75, 1e-6)


class TestProblemCheckTotals(unittest.TestCase):

    def test_double_diamond_model(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()

        top.setup()
        top.run()

        data = top.check_total_derivatives(out_stream=None)
        #print data

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.add('p1', ParamComp('x', 0.5))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        data = top.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_full_model_fd_simple_comp_promoted(self):

        top = Problem()
        top.root = Group()
        sub = top.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        top.root.add('p1', ParamComp('x', 1.0), promotes=['*'])

        top.setup()
        top.run()

        data = top.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()

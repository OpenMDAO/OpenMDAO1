""" Testing for Problem.check_partial_derivatives."""

from six import iteritems
import unittest

import numpy as np

from openmdao.api import Group, Component, IndepVarComp, Problem, ScipyGMRES
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.simple_comps import SimpleArrayComp, SimpleImplicitComp, \
                                      SimpleCompDerivMatVec
from openmdao.test.util import assert_rel_error


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_double_diamond_model_complex_step(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.root.sub1.comp1.fd_options['form'] = 'complex_step'
        prob.root.sub1.sub2.comp2.fd_options['form'] = 'complex_step'
        prob.root.sub1.sub2.comp3.fd_options['form'] = 'complex_step'
        prob.root.sub1.comp4.fd_options['form'] = 'complex_step'
        prob.root.sub3.comp5.fd_options['form'] = 'complex_step'
        prob.root.sub3.comp6.fd_options['form'] = 'complex_step'
        prob.root.comp7.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleArrayComp())
        prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit_complex_step(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.comp.fd_options['step_size'] = 1.0e4
        prob.root.comp.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

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
            def linearize(self, params, unknowns, resids):
                """Analytical derivatives"""
                J = {}
                J[('y', 'x')] = np.zeros((3, 3))
                return J


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BadComp())
        prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        try:
            data = prob.check_partial_derivatives(out_stream=None)
        except Exception as err:
            msg = "derivative in component 'comp' of 'y' wrt 'x' is the wrong size. It should be (2, 2), but got (3, 3)"

            # Good ole Numpy
            raised_error = str(err)
            raised_error = raised_error.replace('3L', '3')

            self.assertEqual(raised_error, msg)
        else:
            self.fail("Error expected")

    def test_big_boy_Jacobian(self):

        class MyComp(Component):

            def __init__(self, multiplier=2.0):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 5.0)

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 3.0*params['x1'] + 4.0*params['x2']

            def linearize(self, params, unknowns, resids):
                """Intentionally left out x2 derivative."""

                J = {}
                J[('y', 'x1')] = np.array([[3.0]])
                return J

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp('x1', 3.0))
        prob.root.add('p2', IndepVarComp('x2', 5.0))

        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)
        self.assertTrue(('y', 'x1') in data['comp'])
        self.assertTrue(('y', 'x2') in data['comp'])

class TestProblemFullFD(unittest.TestCase):

    def test_full_model_fd_simple_comp(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleCompDerivMatVec())
        prob.root.add('p1', IndepVarComp('x', 1.0))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['comp.x']
        unknown_list = ['comp.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['comp.x'][0][0], 2.0, 1e-6)


    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_full_model_fd_double_diamond_grouped(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()
        prob.setup(check=False)
        prob.run()

        prob.root.fd_options['force_fd'] = True

        indep_list = ['sub1.comp1.x1']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)


class TestProblemCheckTotals(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

if __name__ == "__main__":
    unittest.main()

""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.testutil import assert_rel_error

class SrcComp(Component):

    def __init__(self):
        super(SrcComp, self).__init__()

        self.add_param('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x2'] = params['x1']

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x2', 'x1')] = np.array([1.0])
        return J

class TgtCompF(Component):

    def __init__(self):
        super(TgtCompF, self).__init__()

        self.add_param('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J

class TgtCompFMulti(Component):
    # Some extra inputs that might trip things up.

    def __init__(self):
        super(TgtCompFMulti, self).__init__()

        self.add_param('_x2', 100.0, units='degF')
        self.add_param('x2', 100.0, units='degF')
        self.add_param('x2_', 100.0, units='degF')
        self.add_output('_x3', 100.0)
        self.add_output('x3', 100.0)
        self.add_output('x3_', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('_x3', 'x2')] = np.array([1.0])
        J[('_x3', '_x2')] = 0.0
        J[('_x3', 'x2_')] = 0.0
        J[('x3', 'x2')] = np.array([1.0])
        J[('x3', '_x2')] = 0.0
        J[('x3', 'x2_')] = 0.0
        J[('x3_', 'x2')] = np.array([1.0])
        J[('x3_', '_x2')] = 0.0
        J[('x3_', 'x2_')] = 0.0
        return J

class TgtCompC(Component):

    def __init__(self):
        super(TgtCompC, self).__init__()

        self.add_param('x2', 100.0, units='degC')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J

class TgtCompK(Component):

    def __init__(self):
        super(TgtCompK, self).__init__()

        self.add_param('x2', 100.0, units='degK')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[('x3', 'x2')] = np.array([1.0])
        return J


class TestUnitConversion(unittest.TestCase):
    """ Testing automatic unit conversion."""

    def test_basic(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        param_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_input_input(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('tgtC.x2', 'tgtF.x2')
        prob.root.connect('tgtC.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        param_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_implicit_conn(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp(), promotes=['x1', 'x2'])
        prob.root.add('tgtF', TgtCompF(), promotes=['x2'])
        prob.root.add('tgtC', TgtCompC(), promotes=['x2'])
        prob.root.add('tgtK', TgtCompK(), promotes=['x2'])
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        param_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped(self):

        prob = Problem()
        prob.root = Group()
        sub1 = prob.root.add('sub1', Group())
        sub2 = prob.root.add('sub2', Group())
        sub1.add('src', SrcComp())
        sub2.add('tgtF', TgtCompF())
        sub2.add('tgtC', TgtCompC())
        sub2.add('tgtK', TgtCompK())
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')
        prob.root.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.root.connect('sub1.src.x2', 'sub2.tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        # Make sure we don't convert equal units
        self.assertEqual(prob.root.sub2.params.metadata('tgtC.x2').get('unit_conv'),
                         None)

        param_list = ['x1']
        unknown_list = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped_bug_from_pycycle(self):

        prob = Problem()
        root = prob.root = Group()
        sub1 = prob.root.add('sub1', Group(), promotes=['x2'])
        sub1.add('src', SrcComp(), promotes = ['x2'])
        root.add('tgtF', TgtCompFMulti())
        root.add('tgtC', TgtCompC())
        root.add('tgtK', TgtCompK())
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'sub1.src.x1')
        prob.root.connect('x2', 'tgtF.x2')
        prob.root.connect('x2', 'tgtC.x2')
        prob.root.connect('x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        param_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_incompatible_connections(self):

        class BadComp(Component):
            def __init__(self):
                super(BadComp, self).__init__()

                self.add_param('x2', 100.0, units='m')
                self.add_output('x3', 100.0)

        # Explicit Connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('dest', BadComp())
        prob.root.connect('src.x2', 'dest.x2')
        with self.assertRaises(TypeError) as cm:
            prob.setup(check=False)

        expected_msg = "Unit 'degC' in source 'src.x2' is incompatible with unit 'm' in target 'dest.x2'."

        self.assertEqual(str(cm.exception), expected_msg)

        # Implicit Connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp(), promotes=['x2'])
        prob.root.add('dest', BadComp(),promotes=['x2'])
        with self.assertRaises(TypeError) as cm:
            prob.setup(check=False)

        expected_msg = "Unit 'degC' in source 'x2' is incompatible with unit 'm' in target 'x2'."

        self.assertEqual(str(cm.exception), expected_msg)

if __name__ == "__main__":
    unittest.main()

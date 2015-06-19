import unittest
import numpy as np
from openmdao.components.unitcomp import UnitComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.testutil import assert_rel_error

class TestUnitComp(unittest.TestCase):

    def test_instantiation(self):
        u_comp = UnitComp(1, param_name="x", out_name="x_out", units="ft**2/furlong")

        params, unknowns = u_comp._setup_variables()

        self.assertEquals(['x'], list(params.keys()))
        self.assertEquals(['x_out'], list(unknowns.keys()))

    def test_invalid_unit(self):
        prob = Problem()
        prob.root = g = Group()
        g.add('uc', UnitComp(shape=1, param_name='in', out_name='out', units='junk'))
        g.add('pc', ParamComp('x', 0., units='ft'))
        g.connect('pc.x', 'uc.in')
        with self.assertRaises(ValueError) as cm:
            prob.setup()

        expected_msg = "no unit named 'junk' is defined"

        self.assertEqual(expected_msg, str(cm.exception))

    def test_incompatible_units(self):
        prob = Problem()
        prob.root = g = Group()
        g.add('uc', UnitComp(shape=1, param_name='in', out_name='out', units='degC'))
        g.add('pc', ParamComp('x', 0., units='ft'))
        g.connect('pc.x', 'uc.in')
        with self.assertRaises(TypeError) as cm:
            prob.setup()

        expected_msg = "Unit 'ft' in source 'pc.x' is incompatible with unit 'degC' in target 'uc.in'."

        self.assertEqual(expected_msg, str(cm.exception))

    def test_same_name(self):
        with self.assertRaises(ValueError) as cm:
            u = UnitComp(1, 'in', 'in', 'degC')

        expected_msg = "UnitComp param_name cannot match out_name: 'in'"

        self.assertEqual(expected_msg, str(cm.exception))

    def test_values(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('pc', ParamComp('x', 0., units='degC'), promotes=['x'])
        root.add('uc', UnitComp(shape=1, param_name='x', out_name='x_out', units='degF'), promotes=['x', 'x_out'])
        prob.setup()
        prob.run()

        assert_rel_error(self, prob['x_out'], 32., 1e-6)

        param_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'][0][0], 1.8, 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'][0][0], 1.8, 1e-6)

    def test_array_values(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('pc', ParamComp('x', np.zeros((2,3)), units='degC'), promotes=['x'])
        root.add('uc', UnitComp(shape=(2,3), param_name='x', out_name='x_out', units='degF'), promotes=['x', 'x_out'])
        prob.setup()
        prob.run()

        assert_rel_error(self, prob['x_out'], np.array([[32., 32., 32.],[32., 32., 32.]]), 1e-6)

        param_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(param_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

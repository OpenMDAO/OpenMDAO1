import unittest
import numpy as np
from openmdao.api import UnitComp, IndepVarComp, Group, Problem
from openmdao.test.util import assert_rel_error


class TestUnitComp(unittest.TestCase):

    def test_instantiation(self):
        prob = Problem()
        u_comp = UnitComp(1, param_name="x", out_name="x_out", units="ft**2/furlong")

        u_comp._init_sys_data('', prob._probdata)
        params, unknowns = u_comp._setup_variables()

        self.assertEqual(['x'], list(params.keys()))
        self.assertEqual(['x_out'], list(unknowns.keys()))

    def test_invalid_unit(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('uc', UnitComp(shape=1, param_name='in', out_name='out', units='junk'))
        prob.root.add('pc', IndepVarComp('x', 0., units='ft'))
        prob.root.connect('pc.x', 'uc.in')

        with self.assertRaises(ValueError) as cm:
            prob.setup(check=False)

        expected_msg = "no unit named 'junk' is defined"

        self.assertEqual(expected_msg, str(cm.exception))

    def test_incompatible_units(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('uc', UnitComp(shape=1, param_name='in', out_name='out', units='degC'))
        prob.root.add('pc', IndepVarComp('x', 0., units='ft'))
        prob.root.connect('pc.x', 'uc.in')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected_msg = "Unit 'ft' in source 'pc.x' is incompatible with unit 'degC' in target 'uc.in'."

        self.assertTrue(expected_msg in str(cm.exception))

    def test_same_name(self):
        with self.assertRaises(ValueError) as cm:
            u = UnitComp(1, 'in', 'in', 'degC')

        expected_msg = "UnitComp param_name cannot match out_name: 'in'"

        self.assertEqual(expected_msg, str(cm.exception))

    def test_values(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', 0., units='degC'), promotes=['x'])
        prob.root.add('uc', UnitComp(shape=1, param_name='x', out_name='x_out', units='degF'), promotes=['x', 'x_out'])
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x_out'], 32., 1e-6)

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'][0][0], 1.8, 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'][0][0], 1.8, 1e-6)

    def test_array_values(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', np.zeros((2,3)), units='degC'), promotes=['x'])
        prob.root.add('uc', UnitComp(shape=(2,3), param_name='x', out_name='x_out', units='degF'), promotes=['x', 'x_out'])
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x_out'], np.array([[32., 32., 32.],[32., 32., 32.]]), 1e-6)

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

    def test_array_values_same_shape_units(self):

        # Added to improve coverage of an accessor

        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', np.zeros((2, )), units='degC'), promotes=['x'])
        prob.root.add('uc', UnitComp(shape=(2, ), param_name='x', out_name='x_out', units='degF'),
                      promotes=['x', 'x_out'])

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(2), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(2), 1e-6)

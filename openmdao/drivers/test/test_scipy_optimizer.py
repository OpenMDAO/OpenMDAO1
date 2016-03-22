""" Testing optimizer ScipyOptimize."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ScipyOptimizer, ExecComp
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivatives, SellarStateConnection
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error


class TestScipyOptimize(unittest.TestCase):

    def test_simple_paraboloid_unconstrained_TNC(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'TNC'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)
        prob.driver.options['disp'] = False

        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        prob.run()

        # Optimal solution (minimum): x = 6.6667; y = -7.3333
        assert_rel_error(self, prob['x'], 6.666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.333333, 1e-6)

    def test_simple_paraboloid_unconstrained_LBFGSB(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'L-BFGS-B'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Optimal solution (minimum): x = 6.6667; y = -7.3333
        assert_rel_error(self, prob['x'], 6.666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.333333, 1e-6)

    def test_simple_paraboloid_unconstrained_COBYLA(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Optimal solution (minimum): x = 6.6667; y = -7.3333
        assert_rel_error(self, prob['x'], 6.666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.333333, 1e-6)

    def test_simple_paraboloid_unconstrained_SLSQP(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Optimal solution (minimum): x = 6.6667; y = -7.3333
        assert_rel_error(self, prob['x'], 6.666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.333333, 1e-6)

    def test_simple_paraboloid_unconstrained_SLSQP_bounds(self):

        # Make sure we don't go past high/low when set.
        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('obj_comp', ExecComp('obj = -f_xy'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('obj')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x'], 50.0, 1e-6)
        assert_rel_error(self, prob['y'], 50.0, 1e-6)

    def test_simple_paraboloid_constrained_SLSQP(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_constrained_COBYLA_upper(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = y - x'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_constrained_COBYLA_lower(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=15.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', equals=15.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_array_comp_SLSQP(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', equals=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp_SLSQP_scaler_unity_eq(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0, scaler=np.array([1.0, 1.0]))

        prob.driver.add_objective('o', scaler=np.array([1.0, 1.0]))
        prob.driver.add_constraint('c', equals=0.0, scaler=np.array([1.0, 1.0]))
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp_SLSQP_scaler_unity_ineq(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con1', ExecComp('c1 = y - 20.0', c1=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('con2', ExecComp('c2 = y - 20.0', c2=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0, scaler=np.array([1.0, 1.0]))

        prob.driver.add_objective('o', scaler=np.array([1.0, 1.0]))
        prob.driver.add_constraint('c1', lower=0.0, scaler=np.array([1.0, 1.0]))
        prob.driver.add_constraint('c2', upper=0.0, scaler=np.array([1.0, 1.0]))
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros((2, 2))), promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.zeros((2, 2)), y=np.zeros((2, 2))), promotes=['*'])
        root.add('obj', ExecComp('o = y[0, 0]', y=np.zeros((2, 2))), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', equals=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_fan_out(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 1.0))
        root.add('p2', IndepVarComp('x', 1.0))

        root.add('comp1', ExecComp('y = 3.0*x'))
        root.add('comp2', ExecComp('y = 5.0*x'))

        root.add('obj', ExecComp('o = i1 + i2'))
        root.add('con1', ExecComp('c = 15.0 - x'))
        root.add('con2', ExecComp('c = 15.0 - x'))

        # hook up non explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.x', lower=-50.0, upper=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', equals=0.0)
        prob.driver.add_constraint('con2.c', equals=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['obj.o']
        assert_rel_error(self, obj, 30.0, 1e-6)

    def test_Sellar_SLSQP(self):

        prob = Problem()
        prob.root = SellarDerivatives()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                               upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_Sellar_state_SLSQP(self):

        prob = Problem()
        prob.root = SellarStateConnection()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_index_array_param(self):

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8

        prob.driver.add_desvar('z', lower=np.array([-10.0]), upper=np.array([10.0]),
                              indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob['z'][1] = 5.0
        prob.run()

        assert_rel_error(self, prob['z'][0], 0.1005, 1e-3)
        assert_rel_error(self, prob['z'][1], 5.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0, scaler=1/50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

        root.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0, scaler=1/50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

        root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_rev(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0, scaler=1/50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

        root.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fwd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

        root.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_double_sided(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = -x + y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=-11.0, upper=-10.0)

        root.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

        root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_rev(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

        root.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_generate_numpydocstring(self):
        prob = Problem()
        prob.root = SellarStateConnection()
        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8

        prob.driver.add_desvar('z', lower=np.array([-10.0]), upper=np.array([10.0]),
                              indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        test_string = prob.driver.generate_docstring()
        original_string = \
"""    \"\"\"

    Options
    -------
    options['disp'] : bool(False)
        Set to False to prevent printing of Scipy convergence messages
    options['maxiter'] : int(200)
        Maximum number of iterations.
    options['optimizer'] : str('SLSQP')
        Name of optimizer to use
    options['tol'] : float(1e-08)
        Tolerance for termination. For detailed control, use solver-specific options.

    \"\"\"
"""
        for sorig, stest in zip(original_string.split('\n'), test_string.split('\n')):
            self.assertEqual(sorig, stest)

if __name__ == "__main__":
    unittest.main()

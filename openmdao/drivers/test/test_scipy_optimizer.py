""" Testing optimizer ScipyOptimize."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.components.param_comp import ParamComp
from openmdao.components.exec_comp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.drivers.scipy_optimizer import ScipyOptimizer
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivatives, SellarStateConnection
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error


class TestScipyOptimize(unittest.TestCase):

    def test_simple_paraboloid_unconstrained_TNC(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'TNC'
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)
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

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'L-BFGS-B'
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

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

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

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

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

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

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('obj_comp', ExecComp('obj = -f_xy'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

        prob.driver.add_objective('obj')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['x'], 50.0, 1e-6)
        assert_rel_error(self, prob['y'], 50.0, 1e-6)

    def test_simple_paraboloid_constrained_SLSQP(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_constrained_COBYLA(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.add_param('x', low=-50.0, high=50.0)
        prob.driver.add_param('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', ctype='ineq')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_array_comp_SLSQP(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_param('x', low=-50.0, high=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', ctype='eq')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', np.zeros((2, 2))), promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.zeros((2, 2)), y=np.zeros((2, 2))), promotes=['*'])
        root.add('obj', ExecComp('o = y[0, 0]', y=np.zeros((2, 2))), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_param('x', low=-50.0, high=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', ctype='eq')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_fan_out(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('x', 1.0))
        root.add('p2', ParamComp('x', 1.0))

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
        prob.driver.add_param('p1.x', low=-50.0, high=50.0)
        prob.driver.add_param('p2.x', low=-50.0, high=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', ctype='eq')
        prob.driver.add_constraint('con2.c', ctype='eq')
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

        prob.driver.add_param('z', low=np.array([-10.0, 0.0]),
                             high=np.array([10.0, 10.0]))
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
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

        prob.driver.add_param('z', low=np.array([-10.0, 0.0]),
                             high=np.array([10.0, 10.0]))
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
        prob.driver.options['disp'] = False

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

if __name__ == "__main__":
    unittest.main()

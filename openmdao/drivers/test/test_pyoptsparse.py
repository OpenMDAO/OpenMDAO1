""" Testing pyoptsparse SNOPT."""

from pprint import pformat
import os
import unittest

import numpy as np

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.components.exec_comp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error

SKIP = False
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    # Just so python can parse this file.
    from openmdao.core.driver import Driver
    pyOptSparseDriver = Driver
    SKIP = True


class TestPyoptSparse(unittest.TestCase):

    def setUp(self):
        if SKIP is True:
            raise unittest.SkipTest("Could not import pyOptSparseDriver. "
                                    "Is pyoptsparse installed?")

    def tearDown(self):
        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

    def test_simple_paraboloid_upper(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=15.0)

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
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', equals=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_double_sided_low(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=-11.0, upper=-10.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['y'] - prob['x'], -11.0, 1e-6)

    def test_simple_paraboloid_double_sided_high(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0, scaler=1/50.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0, scaler=1/50.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0, scaler=1/50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0, scaler=1/50.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

        root.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_array_comp(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', equals=0.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0, high=50.0)

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', equals=0.0)

        prob.setup(check=False)
        prob.run()

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', np.zeros((2, 2))), promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.zeros((2, 2)), y=np.zeros((2, 2))), promotes=['*'])
        root.add('obj', ExecComp('o = y[0, 0]', y=np.zeros((2, 2))), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('x', low=-50.0*np.ones((2, 2)), high=50.0*np.ones((2, 2)))

        prob.driver.add_objective('o')
        prob.driver.add_constraint('c', equals=0.0)

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

        prob.driver = pyOptSparseDriver()
        prob.driver.add_desvar('p1.x', low=-50.0, high=50.0)
        prob.driver.add_desvar('p2.x', low=-50.0, high=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', equals=0.0)
        prob.driver.add_constraint('con2.c', equals=0.0)

        prob.setup(check=False)
        prob.run()

        obj = prob['obj.o']
        assert_rel_error(self, obj, 30.0, 1e-6)

        # Verify that pyOpt has the correct wrt names
        con1 = prob.driver.pyopt_solution.constraints['con1.c']
        self.assertEqual(con1.wrt, ['p1.x'])
        con2 = prob.driver.pyopt_solution.constraints['con2.c']
        self.assertEqual(con2.wrt, ['p2.x'])

if __name__ == "__main__":
    unittest.main()

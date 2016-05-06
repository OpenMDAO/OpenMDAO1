""" Testing pyoptsparse."""

import os
import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp, Component
from openmdao.core.system import AnalysisError
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error, ConcurrentTestCaseMixin, \
                               set_pyoptsparse_opt


# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class ParaboloidAE(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

    This version raises an analysis error 50% of the time.

    The AE in ParaboloidAE stands for AnalysisError, and is not a
    reference to the Don Bluth film Tian A.E."""

    def __init__(self):
        super(ParaboloidAE, self).__init__()

        self.add_param('x', val=0.0)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.eval_iter_count = 0
        self.eval_fail_at = 3

        self.grad_iter_count = 0
        self.grad_fail_at = 100

        self.fail_hard = False

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """

        if self.eval_iter_count == self.eval_fail_at:
            self.eval_iter_count = 0

            if self.fail_hard:
                raise RuntimeError('This should error.')
            else:
                raise AnalysisError('Try again.')

        x = params['x']
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        self.eval_iter_count += 1

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our paraboloid."""

        if self.grad_iter_count == self.grad_fail_at:
            self.grad_iter_count = 0

            if self.fail_hard:
                raise RuntimeError('This should error.')
            else:
                raise AnalysisError('Try again.')

        x = params['x']
        y = params['y']
        J = {}

        J['f_xy','x'] = 2.0*x - 6.0 + y
        J['f_xy','y'] = 2.0*y + 8.0 + x
        self.grad_iter_count += 1
        return J


class TestPyoptSparse(unittest.TestCase, ConcurrentTestCaseMixin):

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.concurrent_setUp()

    def tearDown(self):
        self.concurrent_tearDown()

    def test_simple_paraboloid_upper(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower_linear(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=15.0, linear=True)
        if OPTIMIZER == 'SNOPT':
            # there is currently a bug in SNOPT, it requires at least one
            # nonlinear inequality constraint, so provide a 'fake' one
            prob.driver.add_constraint('x', lower=-100.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', equals=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality_linear(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', equals=-15.0, linear=True)
        if OPTIMIZER == 'SNOPT':
            # there is currently a bug in SNOPT, it requires at least one
            # nonlinear inequality constraint, so provide a 'fake' one
            prob.driver.add_constraint('x', lower=-100.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
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

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
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

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
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

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
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

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', lower=10.0, upper=11.0, scaler=1/10.)

        root.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_fwd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy', scaler=1/10.)
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

        root.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_rev(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SNOPT':
            prob.driver.opt_settings['Verify level'] = 3
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy', scaler=1/10.)
        prob.driver.add_constraint('c', lower=10.0, upper=11.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)

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
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0*np.ones((2, 2)), upper=50.0*np.ones((2, 2)))

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

        # hook up explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.x', lower=-50.0, upper=50.0)
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

    def test_sparsity_fwd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 1.0))
        root.add('p2', IndepVarComp('x', 1.0))

        root.add('comp1', ExecComp('y = 3.0*x'))
        root.add('comp2', ExecComp('y = 5.0*x'))

        root.add('obj', ExecComp('o = i1 + i2'))
        root.add('con1', ExecComp('c = 15.0 - x'))
        root.add('con2', ExecComp('c = 15.0 - x'))

        # hook up explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.x', lower=-50.0, upper=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', equals=0.0)
        prob.driver.add_constraint('con2.c', equals=0.0)

        prob.setup(check=False)
        prob.run()

        # Verify that the appropriate sparsity pattern is applied
        dv_dict = {'p1.x': 1.0, 'p2.x': 1.0}
        prob.driver._problem = prob
        sens_dict, fail = prob.driver._gradfunc(dv_dict, {})

        self.assertTrue('p2.x' not in sens_dict['con1.c'])
        self.assertTrue('p1.x' in sens_dict['con1.c'])
        self.assertTrue('p2.x' in sens_dict['con2.c'])
        self.assertTrue('p1.x' not in sens_dict['con2.c'])
        self.assertTrue('p1.x' in sens_dict['obj.o'])
        self.assertTrue('p2.x' in sens_dict['obj.o'])

    def test_sparsity_rev(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 1.0))
        root.add('p2', IndepVarComp('x', 1.0))

        root.add('comp1', ExecComp('y = 3.0*x'))
        root.add('comp2', ExecComp('y = 5.0*x'))

        root.add('obj', ExecComp('o = i1 + i2'))
        root.add('con1', ExecComp('c = 15.0 - x'))
        root.add('con2', ExecComp('c = 15.0 - x'))

        # hook up explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.x', lower=-50.0, upper=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', equals=0.0)
        prob.driver.add_constraint('con2.c', equals=0.0)

        prob.root.ln_solver.options['mode'] = 'rev'
        prob.setup(check=False)
        prob.run()

        # Verify that the appropriate sparsity pattern is applied
        dv_dict = {'p1.x': 1.0, 'p2.x': 1.0}
        prob.driver._problem = prob
        sens_dict, fail = prob.driver._gradfunc(dv_dict, {})

        self.assertTrue('p2.x' not in sens_dict['con1.c'])
        self.assertTrue('p1.x' in sens_dict['con1.c'])
        self.assertTrue('p2.x' in sens_dict['con2.c'])
        self.assertTrue('p1.x' not in sens_dict['con2.c'])
        self.assertTrue('p1.x' in sens_dict['obj.o'])
        self.assertTrue('p2.x' in sens_dict['obj.o'])

    def test_sparsity_fd(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 1.0))
        root.add('p2', IndepVarComp('x', 1.0))

        root.add('comp1', ExecComp('y = 3.0*x'))
        root.add('comp2', ExecComp('y = 5.0*x'))

        root.add('obj', ExecComp('o = i1 + i2'))
        root.add('con1', ExecComp('c = 15.0 - x'))
        root.add('con2', ExecComp('c = 15.0 - x'))

        # hook up explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.x', lower=-50.0, upper=50.0)
        prob.driver.add_objective('obj.o')
        prob.driver.add_constraint('con1.c', equals=0.0)
        prob.driver.add_constraint('con2.c', equals=0.0)

        prob.root.fd_options['force_fd'] = True
        prob.setup(check=False)
        prob.run()

        # Verify that the appropriate sparsity pattern is applied
        dv_dict = {'p1.x': 1.0, 'p2.x': 1.0}
        prob.driver._problem = prob
        sens_dict, fail = prob.driver._gradfunc(dv_dict, {})

        self.assertTrue('p2.x' not in sens_dict['con1.c'])
        self.assertTrue('p1.x' in sens_dict['con1.c'])
        self.assertTrue('p2.x' in sens_dict['con2.c'])
        self.assertTrue('p1.x' not in sens_dict['con2.c'])
        self.assertTrue('p1.x' in sens_dict['obj.o'])
        self.assertTrue('p2.x' in sens_dict['obj.o'])

    def test_inf_as_desvar_bounds(self):

        # User may use np.inf as a bound. It is unneccessary, but the user
        # may do it anyway, so make sure SLSQP doesn't blow up with it (bug
        # reported by rfalck)

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-np.inf, upper=np.inf)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_sub_sparsity(self):

        class SegmentComp(Component):
            """
            Each segment accepts y-values of the points within that segments, and computes
            the distance from each point to the origin, and the total area under the segment.
            """

            def __init__(self, M, x0, x1, index):
                super(SegmentComp, self).__init__()
                self.index = index
                self.M = M
                self.x0 = x0
                self.x1 = x1

                self.fd_options['force_fd'] = True

                self.x_i = np.linspace(self.x0, self.x1, M)

                self.add_param(name='y_i', shape=(M, ),
                               desc='y-values of each point in the segment')

                self.add_output(name='r_i', shape=(M, ),
                                desc='distance from each point in the segment to (pi,0)')
                self.add_output(name='area', val=0.0,
                                desc='area under the curve approximated by the trapezodal rule')

            def solve_nonlinear(self, params, unknowns, resids):

                x_i = self.x_i
                y_i = params['y_i']
                unknowns['r_i'] = np.sqrt( x_i**2 + y_i**2 )

                from scipy.integrate import simps
                unknowns['area'] = simps(y_i, x_i)

        class SumAreaComp(Component):
            """
            SumAreaComp takes the area under each segment and sums them to
            compute the total area.
            """

            def __init__(self, N):
                super(SumAreaComp, self).__init__()
                self.N = N
                self.fd_options['force_fd'] = True
                for i in range(N):
                    self.add_param(name='area{0}'.format(i), val=0.0)
                self.add_output(name='area', val=0.0, desc='total area')

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['area'] = 0.0
                for i in range(self.N):
                    unknowns['area'] += params['area{0}'.format(i)]

        N = M = 10
        prob = Problem(root=Group())
        root = prob.root

        b = np.linspace(-np.pi, np.pi, N+1) # segment boundaries

        num_unique_points = N*M-(N-1)

        root.add(name='ivc',
                 system=IndepVarComp(name='y_i', val=np.zeros((num_unique_points, ))),
                 promotes=['y_i'])

        root.add(name='sum_area', system=SumAreaComp(N))

        yi_index = 0
        segments = []
        for i in range(N):
            seg_name = 'seg{0}'.format(i)
            segments.append(SegmentComp(M,b[i], b[i+1], i))
            root.add(name=seg_name, system=segments[i])
            root.connect('{0}.area'.format(seg_name), 'sum_area.area{0}'.format(i))
            yindices = np.arange(yi_index, yi_index+M, dtype=int)
            root.connect('y_i', '{0}.y_i'.format(seg_name), src_indices=yindices)
            yi_index = yi_index+M-1

        prob.driver = driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False

        driver.add_desvar(name="y_i",
                          lower=-100.0,
                          upper=100.0,
                          scaler=1.0,
                          adder=0.0)

        for i in range(N):
            driver.add_constraint(name='seg{0:d}.r_i'.format(i),
                                  lower=0,
                                  upper=np.pi,
                                  scaler=1.0,
                                  adder=0.0)

        driver.add_objective(name='sum_area.area',scaler=-1.0)

        prob.setup(check=False)

        prob['y_i'] = np.linspace(0, 2*np.pi,num_unique_points)

        prob.run()
        assert_rel_error(self, 15.4914, prob['sum_area.area'], 1e-4)

        sub_sparsity = prob.driver.sub_sparsity
        self.assertEquals(len(sub_sparsity['seg0.r_i']['y_i']), 10)

    def test_sub_sparsity_sparse_and_full_in_parallel(self):

        class SegmentComp(Component):
            """
            Each segment accepts y-values of the points within that segments, and computes
            the distance from each point to the origin, and the total area under the segment.
            """

            def __init__(self, M, x0, x1, index):
                super(SegmentComp, self).__init__()
                self.index = index
                self.M = M
                self.x0 = x0
                self.x1 = x1

                self.fd_options['force_fd'] = True

                self.x_i = np.linspace(self.x0, self.x1, M)

                self.add_param(name='y_i', shape=(M, ),
                               desc='y-values of each point in the segment')

                self.add_output(name='r_i', shape=(M, ),
                                desc='distance from each point in the segment to (pi,0)')
                self.add_output(name='area', val=0.0,
                                desc='area under the curve approximated by the trapezodal rule')

            def solve_nonlinear(self, params, unknowns, resids):

                x_i = self.x_i
                y_i = params['y_i']
                unknowns['r_i'] = np.sqrt( x_i**2 + y_i**2 )

                from scipy.integrate import simps
                unknowns['area'] = simps(y_i, x_i)

        class SumAreaComp(Component):
            """
            SumAreaComp takes the area under each segment and sums them to
            compute the total area.
            """

            def __init__(self, N, NNN):
                super(SumAreaComp, self).__init__()
                self.N = N
                self.fd_options['force_fd'] = True
                for i in range(N):
                    self.add_param(name='area{0}'.format(i), val=0.0)
                self.add_output(name='area', val=0.0, desc='total area')
                self.add_output(name='fake', val=0.0, desc='total area')

                self.add_param('bypass', shape=(NNN, ), desc='Param bypass')

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['area'] = 0.0
                for i in range(self.N):
                    unknowns['area'] += params['area{0}'.format(i)]

                unknowns['area'] += np.sum(params['bypass'])

        N = M = 5
        prob = Problem(root=Group())
        root = prob.root

        b = np.linspace(-np.pi, np.pi, N+1) # segment boundaries

        num_unique_points = N*M-(N-1)

        root.add(name='ivc',
                 system=IndepVarComp(name='y_i', val=np.zeros((num_unique_points, ))),
                 promotes=['y_i'])

        root.add(name='sum_area', system=SumAreaComp(N, num_unique_points))

        yi_index = 0
        segments = []
        for i in range(N):
            seg_name = 'seg{0}'.format(i)
            segments.append(SegmentComp(M,b[i], b[i+1], i))
            root.add(name=seg_name, system=segments[i])
            root.connect('{0}.area'.format(seg_name), 'sum_area.area{0}'.format(i))
            yindices = np.arange(yi_index, yi_index+M, dtype=int)
            root.connect('y_i', '{0}.y_i'.format(seg_name), src_indices=yindices)
            yi_index = yi_index+M-1

        root.connect('y_i', 'sum_area.bypass')

        prob.driver = driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False

        driver.add_desvar(name="y_i",
                          lower=-100.0,
                          upper=100.0,
                          scaler=1.0,
                          adder=0.0)

        for i in range(N):
            driver.add_constraint(name='seg{0:d}.r_i'.format(i),
                                  lower=0,
                                  upper=np.pi,
                                  scaler=1.0,
                                  adder=0.0)

        # fake constraint to test the dicts
        driver.add_constraint(name='sum_area.fake', lower=-100.0)

        driver.add_objective(name='sum_area.area',scaler=-1.0)

        prob.setup(check=False)

        prob['y_i'] = np.linspace(0, 2*np.pi,num_unique_points)

        prob.run()

        sub_sparsity = prob.driver.sub_sparsity
        self.assertTrue('sum_area.fake' not in list(prob.driver.sub_sparsity.keys()))


    def test_sub_sparsity_equality_constraints(self):
        # Need this for coverage

        # TODO - Seems to be a Bug in pyoptsparse:SLSQP
        if OPTIMIZER == 'SLSQP':
            raise unittest.SkipTest("SLSQP crashes on this model with eq constraints.")

        class SegmentComp(Component):

            def __init__(self, M, x0, x1, index):
                super(SegmentComp, self).__init__()
                self.index = index
                self.M = M
                self.x0 = x0
                self.x1 = x1

                self.fd_options['force_fd'] = True

                self.x_i = np.linspace(self.x0, self.x1, M)

                self.add_param(name='y_i', shape=(M, ),
                               desc='y-values of each point in the segment')

                self.add_output(name='r_i', shape=(M, ),
                                desc='distance from each point in the segment to (pi,0)')
                self.add_output(name='area', val=0.0,
                                desc='area under the curve approximated by the trapezodal rule')

            def solve_nonlinear(self, params, unknowns, resids):

                x_i = self.x_i
                y_i = params['y_i']
                unknowns['r_i'] = np.sqrt( x_i**2 + y_i**2 )

                from scipy.integrate import simps
                unknowns['area'] = simps(y_i, x_i)

        class SumAreaComp(Component):

            def __init__(self, N):
                super(SumAreaComp, self).__init__()
                self.N = N
                self.fd_options['force_fd'] = True
                for i in range(N):
                    self.add_param(name='area{0}'.format(i), val=0.0)
                self.add_output(name='area', val=0.0, desc='total area')

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['area'] = 0.0
                for i in range(self.N):
                    unknowns['area'] += params['area{0}'.format(i)]

        N = M = 10
        prob = Problem(root=Group())
        root = prob.root

        b = np.linspace(-np.pi, np.pi, N+1) # segment boundaries

        num_unique_points = N*M-(N-1)

        root.add(name='ivc',
                 system=IndepVarComp(name='y_i', val=np.zeros((num_unique_points, ))),
                 promotes=['y_i'])

        root.add(name='sum_area', system=SumAreaComp(N))

        yi_index = 0
        segments = []
        for i in range(N):
            seg_name = 'seg{0}'.format(i)
            segments.append(SegmentComp(M,b[i], b[i+1], i))
            root.add(name=seg_name, system=segments[i])
            root.connect('{0}.area'.format(seg_name), 'sum_area.area{0}'.format(i))
            yindices = np.arange(yi_index, yi_index+M, dtype=int)
            root.connect('y_i', '{0}.y_i'.format(seg_name), src_indices=yindices)
            yi_index = yi_index+M-1

        prob.driver = driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False

        driver.add_desvar(name="y_i",
                          lower=-100.0,
                          upper=100.0,
                          scaler=1.0,
                          adder=0.0)

        for i in range(N):
            driver.add_constraint(name='seg{0:d}.r_i'.format(i),
                                  equals=np.pi)

        driver.add_objective(name='sum_area.area',scaler=-1.0)

        prob.setup(check=False)

        prob['y_i'] = np.linspace(0, 2*np.pi,num_unique_points)

        prob.run()
        assert_rel_error(self, 15.4914, prob['sum_area.area'], 1e-4)

        sub_sparsity = prob.driver.sub_sparsity
        self.assertEquals(len(sub_sparsity['seg0.r_i']['y_i']), 10)

    def test_sub_sparsity_with_desvar_index(self):

        class SegmentComp(Component):
            def __init__(self,M,x0,x1,index):
                super(SegmentComp,self).__init__()
                self.index = index
                self.M = M
                self.x0 = x0
                self.x1 = x1
                self.fd_options['force_fd'] = True
                self.x_i = np.linspace(self.x0,self.x1,M)
                self.add_param(name='y_i',shape=(M,),desc='y-values of each point in the segment')
                self.add_output(name='r_i',shape=(M,),desc='distance from each point in the segment to (pi,0)')
                self.add_output(name='area',val=0.0,desc='area under the curve approximated by the trapezodal rule')

            def solve_nonlinear(self, params, unknowns, resids):
                x_i = self.x_i
                y_i = params['y_i']
                unknowns['r_i'] = np.sqrt( x_i**2 + y_i**2 )
                from scipy.integrate import simps
                unknowns['area'] = simps(y_i,x_i)

        class SumAreaComp(Component):
            def __init__(self,N):
                super(SumAreaComp,self).__init__()
                self.N = N
                self.fd_options['force_fd'] = True
                for i in range(N):
                    self.add_param(name='area{0}'.format(i),val=0.0)
                self.add_output(name='area',val=0.0,desc='total area')

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['area'] = 0.0
                for i in range(self.N):
                    unknowns['area'] += params['area{0}'.format(i)]

        N = M = 10
        prob = Problem(root=Group())
        root = prob.root
        b = np.linspace(-np.pi,np.pi,N+1) # segment boundaries
        num_unique_points = N*M-(N-1)

        root.add(name='ivc',system=IndepVarComp(name='y_i',val=np.zeros((num_unique_points,))),promotes=['y_i'])
        root.add(name='sum_area',system=SumAreaComp(N))

        yi_index = 0
        segments = []
        for i in range(N):
            seg_name = 'seg{0}'.format(i)
            segments.append( SegmentComp(M,b[i],b[i+1],i))
            root.add(name=seg_name,system=segments[i])
            root.connect('{0}.area'.format(seg_name),'sum_area.area{0}'.format(i))
            yindices = np.arange(yi_index,yi_index+M,dtype=int)
            root.connect('y_i','{0}.y_i'.format(seg_name),src_indices=yindices)
            yi_index = yi_index+M-1

        driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False

        prob.driver = driver

        driver.add_desvar(name="y_i",
                          lower=-100.0,
                          upper=100.0,
                          indices=range(1,num_unique_points-1),
                          scaler=1.0,
                          adder=0.0)

        for i in range(N):
            driver.add_constraint(name='seg{0:d}.r_i'.format(i),
                                  lower=0,
                                  upper=np.pi,
                                  scaler=1.0,
                                  adder=0.0)

        driver.add_objective(name='sum_area.area', scaler=-1.0)
        prob.setup(check=False)
        prob['y_i'] = np.zeros(num_unique_points)
        prob.run()
        assert_rel_error(self, 15.4914, prob['sum_area.area'], 1e-4)

        sub_sparsity = prob.driver.sub_sparsity
        self.assertEquals(len(sub_sparsity['seg0.r_i']['y_i']), 9)

    def test_analysis_error_objfunc(self):

        # Component raises an analysis error during some runs, and pyopt
        # attempts to recover.

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidAE(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

        # Normally it takes 9 iterations, but takes 13 here because of the
        # analysis failures. (note SLSQP takes 5 instead of 4)
        if OPTIMIZER == 'SLSQP':
            self.assertEqual(prob.driver.iter_count, 5)
        else:
            self.assertEqual(prob.driver.iter_count, 13)



    def test_raised_error_objfunc(self):

        # Component fails hard this time during execution, so we expect
        # pyoptsparse to raise.


        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidAE(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.root.comp.fail_hard = True

        prob.setup(check=False)

        with self.assertRaises(Exception) as err:
            prob.run()

        # pyopt's failure message differs by platform and is not informative anyway


    def test_analysis_error_sensfunc(self):

        # Component raises an analysis error during some linearize calls, and
        # pyopt attempts to recover.

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidAE(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.root.comp.grad_fail_at = 2
        prob.root.comp.eval_fail_at = 100

        prob.setup(check=False)
        prob.run()

        # SLSQP does a bad job recovering from gradient failures
        if OPTIMIZER == 'SLSQP':
            tol = 1e-2
        else:
            tol = 1e-6

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, tol)
        assert_rel_error(self, prob['y'], -7.833334, tol)

        # Normally it takes 9 iterations, but takes 12 here because of the
        # gradfunc failures. (note SLSQP just doesn't do well)
        if OPTIMIZER == 'SNOPT':
            self.assertEqual(prob.driver.iter_count, 12)


    def test_raised_error_sensfunc(self):

        # Component fails hard this time during gradient eval, so we expect
        # pyoptsparse to raise.

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidAE(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.root.comp.fail_hard = True
        prob.root.comp.grad_fail_at = 2
        prob.root.comp.eval_fail_at = 100

        prob.setup(check=False)

        with self.assertRaises(Exception) as err:
            prob.run()

        # pyopt's failure message differs by platform and is not informative anyway

    def test_pyopt_fd_solution(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', Paraboloid(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        prob.driver.options['gradient method'] = 'pyopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-4)
        assert_rel_error(self, prob['y'], -7.833334, 1e-4)

    def test_pyopt_fd_is_called(self):

        class ParaboloidApplyLinear(Paraboloid):
            def apply_linear(params, unknowns, resids):
                raise Exception("OpenMDAO's finite difference has been called. pyopt_fd\
                                \ option has failed.")

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidApplyLinear(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        prob.driver.options['gradient method'] = 'pyopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        prob.run()


    def test_snopt_fd_solution(self):

        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest()

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', Paraboloid(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        prob.driver.options['gradient method'] = 'snopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_snopt_fd_is_called(self):

        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest()

        class ParaboloidApplyLinear(Paraboloid):
            def apply_linear(params, unknowns, resids):
                raise Exception("OpenMDAO's finite difference has been called. snopt_fd\
                                \ option has failed.")

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', ParaboloidApplyLinear(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER

        prob.driver.options['gradient method'] = 'snopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        prob.run()

    def test_snopt_fd_option_error(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', Paraboloid(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['gradient method'] = 'snopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        with self.assertRaises(Exception) as raises_cm:
            prob.run()

        exception = raises_cm.exception

        msg = "SNOPT's internal finite difference can only be used with SNOPT"

        self.assertEqual(exception.args[0], msg)

    def test_unsupported_multiple_obj(self):
        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])

        root.add('comp', Paraboloid(), promotes=['*'])

        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['gradient method'] = 'snopt_fd'

        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_objective('c')
        prob.driver.add_constraint('c', upper=-15.0)

        expected = 'Multiple objectives have been added to pyOptSparseDriver' \
                   ' but the selected optimizer (SLSQP) does not support' \
                   ' multiple objectives.'

        with self.assertRaises(RuntimeError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected)



if __name__ == "__main__":
    unittest.main()

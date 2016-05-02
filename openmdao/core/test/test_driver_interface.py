""" Test for the Driver class -- basic driver interface."""

from pprint import pformat
import unittest
import warnings

import numpy as np

from openmdao.api import ExecComp, IndepVarComp, Component, Driver, Group, Problem
from openmdao.test.util import assert_rel_error
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simple_comps import ArrayComp2D
from openmdao.test.sellar import SellarDerivatives
from openmdao.util.options import OptionsDictionary
from openmdao.util.record_util import create_local_meta, update_local_meta


class MySimpleDriver(Driver):

    def __init__(self):
        super(MySimpleDriver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['multiple_objectives'] = False

        # My driver options
        self.options = OptionsDictionary()
        self.options.add_option('tol', 1e-4)
        self.options.add_option('maxiter', 10)

        self.alpha = .01
        self.violated = []

    def run(self, problem):
        """ Mimic a very simplistic unconstrained optimization."""

        # Get dicts with pointers to our vectors
        params = self.get_desvars()
        objective = self.get_objectives()
        constraints = self.get_constraints()

        indep_list = params.keys()
        objective_names = list(objective.keys())
        constraint_names = list(constraints.keys())
        unknown_list = objective_names + constraint_names

        itercount = 0
        while itercount < self.options['maxiter']:

            # Run the model
            problem.root.solve_nonlinear()
            #print('z1: %f, z2: %f, x1: %f, y1: %f, y2: %f' % (problem['z'][0],
                                                              #problem['z'][1],
                                                              #problem['x'],
                                                              #problem['y1'],
                                                              #problem['y2']))
            #print('obj: %f, con1: %f, con2: %f' % (problem['obj'], problem['con1'],
                                                   #problem['con2']))

            # Calculate gradient
            J = problem.calc_gradient(indep_list, unknown_list, return_format='dict')

            objective = self.get_objectives()
            constraints = self.get_constraints()

            for key1 in objective_names:
                for key2 in indep_list:

                    grad = J[key1][key2] * objective[key1]
                    new_val = params[key2] - self.alpha*grad

                    # Set parameter
                    self.set_desvar(key2, new_val)

            self.violated = []
            for name, val in constraints.items():
                if np.linalg.norm(val) > 0.0:
                    self.violated.append(name)

            itercount += 1


class Rosenbrock(Component):
    def __init__(self, size=2):
        super(Rosenbrock, self).__init__()
        # self.force_fd = True
        self.add_param('x', val=np.zeros(size))
        self.add_output('f', val=0.0)
        self.add_output('xxx', val=np.zeros(size))

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['f'] = rosen(params['x'])


class ScaleAddDriver(Driver):

    def run(self, problem):
        """ Save away scaled info."""

        self._problem = problem
        self.metadata = create_local_meta(None, 'test')
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        params = self.get_desvars()
        param_meta = self.get_desvar_metadata()

        self.set_desvar('x', 0.5)
        problem.root.solve_nonlinear()

        objective = self.get_objectives()
        constraint = self.get_constraints()

        # Stuff we saved should be in the scaled coordinates.
        self.param = params['x']
        self.obj_scaled = objective['f_xy']
        self.con_scaled = constraint['con']
        self.param_high = param_meta['x']['upper']
        self.param_low = param_meta['x']['lower']


class ScaleAddDriverArray(Driver):

    def run(self, problem):
        """ Save away scaled info."""

        self._problem = problem
        self.metadata = create_local_meta(None, 'test')
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        params = self.get_desvars()
        param_meta = self.get_desvar_metadata()

        self.set_desvar('x', np.array([22.0, 404.0, 9009.0, 121000.0]))
        problem.root.solve_nonlinear()

        objective = self.get_objectives()
        constraint = self.get_constraints()

        # Stuff we saved should be in the scaled coordinates.
        self.param = params['x']
        self.obj_scaled = objective['y']
        self.con_scaled = constraint['con']
        self.param_low = param_meta['x']['lower']


class TestDriver(unittest.TestCase):

    def test_mydriver(self):

        prob = Problem()
        root = prob.root = SellarDerivatives()

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('z', lower=-100.0, upper=100.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.setup(check=False)
        prob.run()

        obj = prob['obj']
        self.assertLess(obj, 28.0)

    def test_scaler_adder(self):

        prob = Problem()
        root = prob.root = Group()
        driver = prob.driver = ScaleAddDriver()

        root.add('p1', IndepVarComp([('x',60000.0,{'desc':'my x'}),
                                  ('y',60000.0,{'desc':'my y'})]), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('constraint', ExecComp('con=f_xy + x + y'), promotes=['*'])

        driver.add_desvar('x', lower=59000.0, upper=61000.0, adder=-60000.0, scaler=1/1000.0)
        driver.add_objective('f_xy', adder=-10890367002.0, scaler=1.0/20)
        driver.add_constraint('con', upper=0.0, adder=-10890487502.0, scaler=1.0/20)

        prob.setup(check=False)
        prob.run()

        self.assertEqual(driver.param_high, 1.0)
        self.assertEqual(driver.param_low, -1.0)
        self.assertEqual(driver.param, 0.0)
        self.assertEqual(prob['x'], 60500.0)
        self.assertEqual(driver.obj_scaled[0], 1.0)
        self.assertEqual(driver.con_scaled[0], 1.0)

    def test_scaler_adder_int(self):

        prob = Problem()
        root = prob.root = Group()
        driver = prob.driver = ScaleAddDriver()

        root.add('p1', IndepVarComp([('x',12.0,{'desc':'my x'}),
                                     ('y',13.0,{'desc':'my y'})]), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('constraint', ExecComp('con=f_xy + x + y'), promotes=['*'])

        driver.add_desvar('x', adder=-10, scaler=20.0)
        driver.add_objective('f_xy', adder=-10, scaler=20)
        driver.add_constraint('con', upper=0, adder=-10, scaler=20)

        prob.setup(check=False)
        prob.run()

        self.assertEqual(driver.param, 40.0)
        self.assertEqual(prob['x'], 10.025)
        assert_rel_error(self, driver.obj_scaled[0], 9113.5125, 1e-6)
        assert_rel_error(self, driver.con_scaled[0], 9574.0125, 1e-6)

        J = driver.calc_gradient(['x', 'y'], ['f_xy'])
        assert_rel_error(self, J[0][0], 27.05, 1e-6)
        assert_rel_error(self, J[0][1], 880.5, 1e-6)

    def test_scaler_adder_array(self):

        prob = Problem()
        root = prob.root = Group()
        driver = prob.driver = ScaleAddDriverArray()

        root.add('p1', IndepVarComp('x', val=np.array([[1.0, 1.0], [1.0, 1.0]])),
                 promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('constraint', ExecComp('con = x + y',
                                        x=np.array([[1.0, 1.0], [1.0, 1.0]]),
                                        y=np.array([[1.0, 1.0], [1.0, 1.0]]),
                                        con=np.array([[1.0, 1.0], [1.0, 1.0]])),
                 promotes=['*'])

        driver.add_desvar('x', lower=np.array([[-1e5, -1e5], [-1e5, -1e5]]),
                          upper=np.array([1e25, 1e25, 1e25, 1e25]),
                         adder=np.array([[10.0, 100.0], [1000.0, 10000.0]]),
                         scaler=np.array([[1.0, 2.0], [3.0, 4.0]]))
        driver.add_objective('y', adder=np.array([[10.0, 100.0], [1000.0, 10000.0]]),
                         scaler=np.array([[1.0, 2.0], [3.0, 4.0]]))
        driver.add_constraint('con', upper=np.zeros((2, 2)), adder=np.array([[10.0, 100.0], [1000.0, 10000.0]]),
                              scaler=np.array([[1.0, 2.0], [3.0, 4.0]]))

        prob.setup(check=False)
        prob.run()

        self.assertEqual(driver.param[0], 11.0)
        self.assertEqual(driver.param[1], 202.0)
        self.assertEqual(driver.param[2], 3003.0)
        self.assertEqual(driver.param[3], 40004.0)
        self.assertEqual(prob['x'][0, 0], 12.0)
        self.assertEqual(prob['x'][0, 1], 102.0)
        self.assertEqual(prob['x'][1, 0], 2003.0)
        self.assertEqual(prob['x'][1, 1], 20250.0)
        self.assertEqual(driver.obj_scaled[0], (prob['y'][0, 0] + 10.0)*1.0)
        self.assertEqual(driver.obj_scaled[1], (prob['y'][0, 1] + 100.0)*2.0)
        self.assertEqual(driver.obj_scaled[2], (prob['y'][1, 0] + 1000.0)*3.0)
        self.assertEqual(driver.obj_scaled[3], (prob['y'][1, 1] + 10000.0)*4.0)
        self.assertEqual(driver.param_low[0], (-1e5 + 10.0)*1.0)
        self.assertEqual(driver.param_low[1], (-1e5 + 100.0)*2.0)
        self.assertEqual(driver.param_low[2], (-1e5 + 1000.0)*3.0)
        self.assertEqual(driver.param_low[3], (-1e5 + 10000.0)*4.0)
        conval = prob['x'] + prob['y']
        self.assertEqual(driver.con_scaled[0], (conval[0, 0] + 10.0)*1.0)
        self.assertEqual(driver.con_scaled[1], (conval[0, 1] + 100.0)*2.0)
        self.assertEqual(driver.con_scaled[2], (conval[1, 0] + 1000.0)*3.0)
        self.assertEqual(driver.con_scaled[3], (conval[1, 1] + 10000.0)*4.0)

    def test_scaler_adder_array_int(self):

        prob = Problem()
        root = prob.root = Group()
        driver = prob.driver = ScaleAddDriverArray()

        root.add('p1', IndepVarComp('x', val=np.array([[1.0, 1.0], [1.0, 1.0]])),
                 promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('constraint', ExecComp('con = x + y',
                                        x=np.array([[1.0, 1.0], [1.0, 1.0]]),
                                        y=np.array([[1.0, 1.0], [1.0, 1.0]]),
                                        con=np.array([[1.0, 1.0], [1.0, 1.0]])),
                 promotes=['*'])

        driver.add_desvar('x', lower=np.array([[-1e5, -1e5], [-1e5, -1e5]]),
                          adder=np.array([[10, 100], [1000, 10000]]),
                          scaler=np.array([[1, 2], [3, 4]]))
        driver.add_objective('y', adder=np.array([[10, 100], [1000, 10000]]),
                             scaler=np.array([[1, 2], [3, 4]]))
        driver.add_constraint('con', upper=np.zeros((2, 2)), adder=np.array([[10, 100], [1000, 10000]]),
                              scaler=np.array([[1, 2], [3, 4]]))

        prob.setup(check=False)
        prob.run()

        self.assertEqual(driver.param[0], 11.0)
        self.assertEqual(driver.param[1], 202.0)
        self.assertEqual(driver.param[2], 3003.0)
        self.assertEqual(driver.param[3], 40004.0)
        self.assertEqual(prob['x'][0, 0], 12.0)
        self.assertEqual(prob['x'][0, 1], 102.0)
        self.assertEqual(prob['x'][1, 0], 2003.0)
        self.assertEqual(prob['x'][1, 1], 20250.0)
        self.assertEqual(driver.obj_scaled[0], (prob['y'][0, 0] + 10.0)*1.0)
        self.assertEqual(driver.obj_scaled[1], (prob['y'][0, 1] + 100.0)*2.0)
        self.assertEqual(driver.obj_scaled[2], (prob['y'][1, 0] + 1000.0)*3.0)
        self.assertEqual(driver.obj_scaled[3], (prob['y'][1, 1] + 10000.0)*4.0)
        self.assertEqual(driver.param_low[0], (-1e5 + 10.0)*1.0)
        self.assertEqual(driver.param_low[1], (-1e5 + 100.0)*2.0)
        self.assertEqual(driver.param_low[2], (-1e5 + 1000.0)*3.0)
        self.assertEqual(driver.param_low[3], (-1e5 + 10000.0)*4.0)
        conval = prob['x'] + prob['y']
        self.assertEqual(driver.con_scaled[0], (conval[0, 0] + 10.0)*1.0)
        self.assertEqual(driver.con_scaled[1], (conval[0, 1] + 100.0)*2.0)
        self.assertEqual(driver.con_scaled[2], (conval[1, 0] + 1000.0)*3.0)
        self.assertEqual(driver.con_scaled[3], (conval[1, 1] + 10000.0)*4.0)

        J = driver.calc_gradient(['x'], ['y', 'con'])
        Jbase = np.array([[  2.,   1.,   3.,   7.],
                          [  4.,   2.,   6.,   5.],
                          [  3.,   6.,   9.,   8.],
                          [  1.,   3.,   2.,   4.],
                          [  3.,   1.,   3.,   7.],
                          [  4.,   3.,   6.,   5.],
                          [  3.,   6.,  10.,   8.],
                          [  1.,   3.,   2.,   5.]])
        assert_rel_error(self, J, Jbase, 1e-6)

    def test_eq_ineq_error_messages(self):

        prob = Problem()
        root = prob.root = SellarDerivatives()

        prob.driver = MySimpleDriver()

        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1')

        self.assertEqual(str(cm.exception), "Constraint 'con1' needs to define lower, upper, or equals.")

        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1', lower=0.0, upper=1.1, equals=2.2)

        self.assertEqual(str(cm.exception), "Constraint 'con1' cannot be both equality and inequality.")

        # Don't try this at home, kids
        prob.driver.supports['two_sided_constraints'] = False

        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1', lower=0.0, upper=1.1)

        self.assertEqual(str(cm.exception), "Driver does not support 2-sided constraint 'con1'.")

        # Don't try this at home, kids
        prob.driver.supports['equality_constraints'] = False

        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1', equals=0.0)

        self.assertEqual(str(cm.exception), "Driver does not support equality constraint 'con1'.")

        # Don't try this at home, kids
        prob.driver.supports['inequality_constraints'] = False

        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1', upper=0.0)

        self.assertEqual(str(cm.exception), "Driver does not support inequality constraint 'con1'.")

    def test_index_error_messages_param(self):

        prob = Problem()
        prob.root = Group()
        prob.root.fd_options['force_fd'] = True
        prob.root.ln_solver.options['mode'] = 'auto'

        prob.root.add('myparams', IndepVarComp('x', np.zeros(4)))
        prob.root.add('rosen', Rosenbrock(4))

        prob.root.connect('myparams.x', 'rosen.x')

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('myparams.x', indices=[0, 3, 4])
        prob.driver.add_objective('rosen.f')

        prob.setup(check=False)

        # Make sure we can't do this
        with self.assertRaises(IndexError) as cm:
            prob.run()

        msg = "Index for design var 'myparams.x' is out of bounds. "
        msg += "Requested index: [0 3 4], "
        msg += "shape: (4,)."
        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(4L,', '(4,')
        self.assertEqual(msg, raised_error)

    def test_index_error_messages_obj(self):

        prob = Problem()
        prob.root = Group()
        prob.root.fd_options['force_fd'] = True
        prob.root.ln_solver.options['mode'] = 'auto'

        prob.root.add('myparams', IndepVarComp('x', np.zeros(4)))
        prob.root.add('rosen', Rosenbrock(4))

        prob.root.connect('myparams.x', 'rosen.x')

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('myparams.x')
        prob.driver.add_objective('rosen.xxx', indices=[4])

        prob.setup(check=False)

        # Make sure we can't do this
        with self.assertRaises(IndexError) as cm:
            prob.run()

        msg = "Index for objective 'rosen.xxx' is out of bounds. "
        msg += "Requested index: [4], "
        msg += "shape: (4,)."
        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(4L,', '(4,')
        self.assertEqual(msg, raised_error)

    def test_index_error_messages_con(self):

        prob = Problem()
        prob.root = Group()
        prob.root.fd_options['force_fd'] = True
        prob.root.ln_solver.options['mode'] = 'auto'

        prob.root.add('myparams', IndepVarComp('x', np.zeros(4)))
        prob.root.add('rosen', Rosenbrock(4))

        prob.root.connect('myparams.x', 'rosen.x')

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('myparams.x')
        prob.driver.add_constraint('rosen.xxx', upper=0.0, indices=[4])

        prob.setup(check=False)

        # Make sure we can't do this
        with self.assertRaises(IndexError) as cm:
            prob.run()

        msg = "Index for constraint 'rosen.xxx' is out of bounds. "
        msg += "Requested index: [4], "
        msg += "shape: (4,)."
        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(4L,', '(4,')
        self.assertEqual(msg, raised_error)

    def test_add_duplicate(self):

        prob = Problem()
        root = prob.root = SellarDerivatives()

        prob.driver = MySimpleDriver()

        # For this test only assume the driver supports multiple objectives
        prob.driver.supports['multiple_objectives'] = True


        prob.driver.add_desvar('z', lower=-100.0, upper=100.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)

        # Add duplicate desvar
        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_desvar('z', lower=-50.0, upper=49.0)

        msg = "Desvar 'z' already exists."
        raised_error = str(cm.exception)
        self.assertEqual(msg, raised_error)

        # Add duplicate constraint
        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_constraint('con1', upper=0.0)

        msg = "Constraint 'con1' already exists."
        raised_error = str(cm.exception)
        self.assertEqual(msg, raised_error)

        # Add duplicate objective
        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_objective('obj')

        msg = "Objective 'obj' already exists."
        raised_error = str(cm.exception)
        self.assertEqual(msg, raised_error)

    def test_unsupported_multiple_obj(self):
        prob = Problem()
        prob.root = SellarDerivatives()

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('z', lower=-100.0, upper=100.0)

        prob.driver.add_objective('obj')

        # Add duplicate objective
        with self.assertRaises(RuntimeError) as cm:
            prob.driver.add_objective('x')

        msg = "Attempted to add multiple objectives to a driver that does not " \
              "support multiple objectives."
        raised_error = str(cm.exception)
        self.assertEqual(msg, raised_error)

    def test_no_desvar_bound(self):

        prob = Problem()
        root = prob.root = SellarDerivatives()

        prob.driver = MySimpleDriver()
        prob.driver.add_desvar('z')

        prob.setup(check=False)

        meta = prob.driver._desvars['z']
        self.assertLess(meta['lower'], -1e12)
        self.assertGreater(meta['upper'], 1e12)


class TestDeprecated(unittest.TestCase):
    def test_deprecated_add_param(self):
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            p = Problem()
            p.driver.add_param('x', 1.0)

            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message),
                             'Driver.add_param() is deprecated. Use add_desvar() instead.')

if __name__ == "__main__":
    unittest.main()

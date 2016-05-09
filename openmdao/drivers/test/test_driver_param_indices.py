""" Testing desvar indices, param indices, and associated derivatives.."""

import os

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer, \
    LinearGaussSeidel
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.util import assert_rel_error, set_pyoptsparse_opt
from openmdao.util.options import OptionsDictionary

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestParamIndicesScipy(unittest.TestCase):

    def test_Sellar_state_SLSQP(self):
        """ Baseline Sellar test case without specifying indices.
        """

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

    def test_driver_param_indices_slsqp(self):
        """ Test driver param indices with ScipyOptimizer SLSQP and force_fd=False
        """

        prob = Problem()
        prob.root = SellarStateConnection()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = False
        prob.driver.options['tol'] = 1.0e-8
        prob.root.fd_options['force_fd'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0]),
                                    upper=np.array([10.0]), indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_driver_param_indices_slsqp_force_fd(self):
        """ Test driver param indices with ScipyOptimizer SLSQP and force_fd=True
        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = True

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0]),
                                    upper=np.array([10.0]), indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)


class TestParamIndicesPyoptsparse(unittest.TestCase):

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

    def tearDown(self):
        try:
            os.remove('SLSQP.out')
        except OSError:
            pass

        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

    def test_driver_param_indices(self):
        """ Test driver param indices with pyOptSparse and force_fd=False
        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = False

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0]),
                                    upper=np.array([10.0]), indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_driver_param_indices_force_fd(self):
        """ Test driver param indices with pyOptSparse and force_fd=True
        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = True

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0]),
                                    upper=np.array([10.0]), indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_driver_param_indices_force_fd_shift(self):
        """ Test driver param indices with shifted indices and force_fd=True
        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = True

        prob.driver.add_desvar('z', lower=np.array([-10.0, -10.0]),
                                    upper=np.array([10.0, 10.0]), indices=[1])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        J = prob.calc_gradient(['x', 'z'], ['obj'], mode='fd',
                               return_format='array')
        assert_rel_error(self, J[0][1], 1.78402, 1e-3)

    def test_poi_index_w_irrelevant_var(self):
        prob = Problem()
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.root = root = Group()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        root.add('p1', IndepVarComp('x', np.array([1.0, 3.0, 4.0])))
        root.add('p2', IndepVarComp('x', np.array([5.0, 2.0, -1.0])))
        root.add('C1', ExecComp('y = 2.0*x', x=np.zeros(3), y=np.zeros(3)))
        root.add('C2', ExecComp('y = 3.0*x', x=np.zeros(3), y=np.zeros(3)))
        root.add('con1', ExecComp('c = 7.0 - y', y=np.zeros(3), c=np.zeros(3)))
        root.add('con2', ExecComp('c = 2.0 - y', y=np.zeros(3), c=np.zeros(3)))
        root.add('obj', ExecComp('o = y1+y2'))

        prob.driver.add_desvar('p1.x', indices=[1])
        prob.driver.add_desvar('p2.x', indices=[2])
        prob.driver.add_constraint('con1.c', upper=0.0, indices=[1])
        prob.driver.add_constraint('con2.c', upper=0.0, indices=[2])
        prob.driver.add_objective('obj.o')

        root.connect('p1.x', 'C1.x')
        root.connect('p2.x', 'C2.x')
        root.connect('C1.y', 'con1.y')
        root.connect('C2.y', 'con2.y')
        root.connect('C1.y', 'obj.y1', src_indices=[1])
        root.connect('C2.y', 'obj.y2', src_indices=[2])

        prob.root.ln_solver.options['mode'] = 'rev'
        prob.setup(check=False)
        prob.run()

        # I was trying in this test to duplicate an error in pointer, but wasn't able to.
        # I was able to find a different error that occurred when using return_format='array'
        # that was also fixed by the same PR that fixed pointer.
        J = prob.calc_gradient(['p1.x', 'p2.x'], ['con1.c', 'con2.c'], mode='rev',
                               return_format='array')

        assert_rel_error(self, J[0][0], -2.0, 1e-3)
        assert_rel_error(self, J[0][1], .0, 1e-3)
        assert_rel_error(self, J[1][0], .0, 1e-3)
        assert_rel_error(self, J[1][1], -3.0, 1e-3)

        J = prob.calc_gradient(['p1.x', 'p2.x'], ['con1.c', 'con2.c'], mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['con1.c']['p1.x'], -2.0, 1e-3)
        assert_rel_error(self, J['con1.c']['p2.x'], .0, 1e-3)
        assert_rel_error(self, J['con2.c']['p1.x'], .0, 1e-3)
        assert_rel_error(self, J['con2.c']['p2.x'], -3.0, 1e-3)

        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x', 'p2.x'], ['con1.c', 'con2.c'], mode='fwd',
                               return_format='array')

        assert_rel_error(self, J[0][0], -2.0, 1e-3)
        assert_rel_error(self, J[0][1], .0, 1e-3)
        assert_rel_error(self, J[1][0], .0, 1e-3)
        assert_rel_error(self, J[1][1], -3.0, 1e-3)

        J = prob.calc_gradient(['p1.x', 'p2.x'], ['con1.c', 'con2.c'], mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['con1.c']['p1.x'], -2.0, 1e-3)
        assert_rel_error(self, J['con1.c']['p2.x'], .0, 1e-3)
        assert_rel_error(self, J['con2.c']['p1.x'], .0, 1e-3)
        assert_rel_error(self, J['con2.c']['p2.x'], -3.0, 1e-3)


class TestMiscParamIndices(unittest.TestCase):

    def test_param_as_obj_scaler_explicit(self):

        prob = Problem()
        root = prob.root = Group()
        root.add('comp', ExecComp('y = 3.0*x'))
        root.add('p', IndepVarComp('x', 3.0))
        root.connect('p.x', 'comp.x')

        prob.driver.add_desvar('p.x', 1.0)
        prob.driver.add_objective('p.x')
        prob.driver.add_constraint('comp.y', lower=-100.0)

        prob.setup(check=False)

        # Cheat to make Driver give derivs
        prob.driver._problem = prob

        prob.run()

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fwd', return_format='dict')
        self.assertEqual(J['p.x']['p.x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fwd', return_format='array')
        self.assertEqual(J[0][0], 1.0)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='rev', return_format='dict')
        self.assertEqual(J['p.x']['p.x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='rev', return_format='array')
        self.assertEqual(J[0][0], 1.0)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fd', return_format='dict')
        self.assertEqual(J['p.x']['p.x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fd', return_format='array')
        self.assertEqual(J[0][0], 1.0)

    def test_param_as_obj_scaler_implicit(self):

        prob = Problem()
        root = prob.root = Group()
        root.add('comp', ExecComp('y = 3.0*x'), promotes=['x', 'y'])
        root.add('p', IndepVarComp('x', 3.0), promotes=['x'])

        prob.driver.add_desvar('x', 1.0)
        prob.driver.add_objective('x')
        prob.driver.add_constraint('y', lower=-100.0)

        prob.setup(check=False)

        # Cheat to make Driver give derivs
        prob.driver._problem = prob

        prob.run()

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fwd', return_format='dict')
        self.assertEqual(J['x']['x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fwd', return_format='array')
        self.assertEqual(J[0][0], 1.0)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='rev', return_format='dict')
        self.assertEqual(J['x']['x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='rev', return_format='array')
        self.assertEqual(J[0][0], 1.0)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fd', return_format='dict')
        self.assertEqual(J['x']['x'][0][0], 1.0)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fd', return_format='array')
        self.assertEqual(J[0][0], 1.0)

    def test_param_as_obj_1darray_explicit(self):

        prob = Problem()
        root = prob.root = Group()
        root.add('comp', ExecComp('y = 3.0*x', x=np.zeros((10, )), y=np.zeros((10, )) ))
        root.add('p', IndepVarComp('x', np.zeros((10, )) ))
        root.connect('p.x', 'comp.x')

        prob.driver.add_desvar('p.x', np.ones((8, )), indices=[1, 2, 3, 4, 5, 6, 7, 8])
        prob.driver.add_objective('p.x', indices=[5, 6, 7])
        prob.driver.add_constraint('comp.y', lower=-100.0)

        prob.setup(check=False)

        # Cheat to make Driver give derivs
        prob.driver._problem = prob

        prob.run()

        Jbase = np.zeros((3, 8))
        Jbase[0, 4] = 1.0
        Jbase[1, 5] = 1.0
        Jbase[2, 6] = 1.0

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fwd', return_format='dict')
        diff = np.linalg.norm(J['p.x']['p.x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fwd', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['p.x']['p.x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='rev', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fd', return_format='dict')
        diff = np.linalg.norm(J['p.x']['p.x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['p.x'], ['p.x'], mode='fd', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

    def test_param_as_obj_1darray_implicit(self):

        prob = Problem()
        root = prob.root = Group()
        root.add('comp', ExecComp('y = 3.0*x', x=np.zeros((10, )), y=np.zeros((10, )) ),
                 promotes=['x', 'y'])
        root.add('p', IndepVarComp('x', np.zeros((10, )) ), promotes=['x'])

        prob.driver.add_desvar('x', np.ones((8, )), indices=[1, 2, 3, 4, 5, 6, 7, 8])
        prob.driver.add_objective('x', indices=[5, 6, 7])
        prob.driver.add_constraint('y', lower=-100.0)

        prob.setup(check=False)

        # Cheat to make Driver give derivs
        prob.driver._problem = prob

        prob.run()

        Jbase = np.zeros((3, 8))
        Jbase[0, 4] = 1.0
        Jbase[1, 5] = 1.0
        Jbase[2, 6] = 1.0

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fwd', return_format='dict')
        diff = np.linalg.norm(J['x']['x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fwd', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['x']['x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='rev', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fd', return_format='dict')
        diff = np.linalg.norm(J['x']['x'] - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

        J = prob.driver.calc_gradient(['x'], ['x'], mode='fd', return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1.0e-9)

if __name__ == "__main__":
    unittest.main()

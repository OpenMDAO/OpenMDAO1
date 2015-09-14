""" Testing optimizer ScipyOptimize."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.components.param_comp import ParamComp
from openmdao.components.exec_comp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.drivers import ScipyOptimizer
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivatives, SellarStateConnection
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


class TestParamIndices(unittest.TestCase):

    def setUp(self):
        if SKIP is True:
            raise unittest.SkipTest("Could not import pyOptSparseDriver. "
                                    "Is pyoptsparse installed?")


    def test_Sellar_state_SLSQP(self):
        """ Baseline Sellar test case without specifying indices.
        """

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


    def test_driver_param_indices_slsqp(self):
        """ Test driver param indices with ScipyOptimizer SLSQP and force_fd=False

        """

        prob = Problem()
        prob.root = SellarStateConnection()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.root.fd_options['force_fd'] = False

        prob.driver.add_param('z', low=np.array([-10.0]),
                              high=np.array([10.0]),indices=[0])
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
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

        prob.driver.add_param('z', low=np.array([-10.0]),
                              high=np.array([10.0]),indices=[0])
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)


    def test_driver_param_indices_snopt(self):
        """ Test driver param indices with pyOptSparse and force_fd=False

        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = False

        prob.driver = pyOptSparseDriver()

        prob.driver.add_param('z', low=np.array([-10.0]),
                              high=np.array([10.0]),indices=[0])
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)


    def test_driver_param_indices_snopt_force_fd(self):
        """ Testt driver param indices with pyOptSparse and force_fd=True

        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_param('z', low=np.array([-10.0]),
                              high=np.array([10.0]), indices=[0])
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_driver_param_indices_snopt_force_fd_shift(self):
        """ Testt driver param indices with pyOptSparse and force_fd=True

        """

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.fd_options['force_fd'] = True

        prob.driver.add_param('z', low=np.array([-10.0, -10.0]),
                              high=np.array([10.0, 10.0]), indices=[1])
        prob.driver.add_param('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')
        #prob.driver.options['disp'] = False

        prob.setup(check=False)

        prob['z'][1] = 0.0

        prob.run()

        J = prob.calc_gradient(['x', 'z'], ['obj'], mode='fd',
                              return_format='array')
        assert_rel_error(self, J[0][1], 1.78402, 1e-3)



if __name__ == "__main__":
    unittest.main()

""" Unit test for the InMemoryRecorder. This recorder is mostly used for testing
anyway. """

import unittest

import numpy as np

from openmdao.api import Problem, InMemoryRecorder, ScipyOptimizer
from openmdao.test.sellar import SellarDerivativesGrouped
from openmdao.test.util import assert_rel_error

# check that pyoptsparse is installed
# if it is, try to use SLSQP
OPT = None
OPTIMIZER = None

try:
    from pyoptsparse import OPT
    try:
        OPT('SLSQP')
        OPTIMIZER = 'SLSQP'
    except:
        pass
except:
    pass

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestInMemoryRecorder(unittest.TestCase):

    def setUp(self):
        self.recorder = InMemoryRecorder()

    def test_root_derivs_dict(self):

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        self.recorder.options['record_derivs'] = True
        prob.setup(check=False)

        prob.run()

        prob.cleanup()

        J1 = self.recorder.deriv_iters[0]['Derivatives']

        Jbase = {}
        Jbase['con1'] = {}
        Jbase['con1']['x'] = -0.98061433
        Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
        Jbase['con2'] = {}
        Jbase['con2']['x'] = 0.09692762
        Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
        Jbase['obj'] = {}
        Jbase['obj']['x'] = 2.98061392
        Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J1[key1][key2], val2, .00001)

    def test_root_derivs_array(self):
        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        self.recorder.options['record_derivs'] = True
        prob.setup(check=False)

        prob.run()

        prob.cleanup()

        J1 = self.recorder.deriv_iters[0]['Derivatives']

        assert_rel_error(self, J1[0][0], 9.61001155, .00001)
        assert_rel_error(self, J1[0][1], 1.78448534, .00001)
        assert_rel_error(self, J1[0][2], 2.98061392, .00001)
        assert_rel_error(self, J1[1][0], -9.61002285, .00001)
        assert_rel_error(self, J1[1][1], -0.78449158, .00001)
        assert_rel_error(self, J1[1][2], -0.98061433, .00001)
        assert_rel_error(self, J1[2][0], 1.94989079, .00001)
        assert_rel_error(self, J1[2][1], 1.0775421, .00001)
        assert_rel_error(self, J1[2][2], 0.09692762, .00001)

if __name__ == "__main__":
    unittest.main()

""" Testing doe drivers with array for design variables """

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, Component

from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver
from openmdao.drivers.fullfactorial_driver import FullFactorialDriver
from openmdao.drivers.uniform_driver import UniformDriver
from openmdao.test.util import assert_rel_error

SEED = 0
LHS1 = [[-11.2796624 ,  17.87973416,  -9.9309156 ],
        [-36.37792043, -14.40863002,  41.14735283],
        [ 10.93968028,  47.29432502,  24.09156901],
        [ 34.58603797, -30.20687405, -36.77762701]]
FACT1 = [[-50., -50., -50.],
         [-50., -50.,  50.],
         [-50.,  50., -50.],
         [-50.,  50.,  50.],
         [ 50., -50., -50.],
         [ 50., -50.,  50.],
         [ 50.,  50., -50.],
         [ 50.,  50.,  50.]]
UNIFORM1 = [[  4.88135039,  21.51893664,  10.27633761],
            [  4.4883183 ,  -7.63452007,  14.58941131],
            [ -6.24127887,  39.17730008,  46.36627605],
            [-11.65584812,  29.17250381,   2.88949198]]

LHS2 = [([-11.2796624] , 17.879734159310487 , [-9.9309156]),
        ([-36.37792043], -14.408630016527383, [ 41.14735283]),
        ([ 10.93968028], 47.294325019551991 , [ 24.09156901]),
        ([ 34.58603797], -30.206874047933386, [-36.77762701])]
FACT2 = [([-50.], -50.0, [-50.]),
         ([-50.], -50.0, [ 50.]),
         ([-50.], 50.0, [-50.]),
         ([-50.], 50.0, [ 50.]),
         ([ 50.], -50.0, [-50.]),
         ([ 50.], -50.0, [ 50.]),
         ([ 50.], 50.0, [-50.]),
         ([ 50.], 50.0, [ 50.])]
UNIFORM2 = [([ 4.88135039] , 21.51893663724195,   [ 10.27633761]),
            ([ 4.4883183]  , -7.6345200661095305, [ 14.58941131]),
            ([-6.24127887] , 39.177300078207978,  [ 46.36627605]),
            ([-11.65584812], 29.172503808266455,  [ 2.88949198])]


class FakeCompWithArrayParam(Component):
    def __init__(self):
        super(FakeCompWithArrayParam, self).__init__()
        self.add_param('X', val=np.array([0., 0., 0.]))
        self.add_output('f_xy', val=0.0)
        self.callargs = []

    def solve_nonlinear(self, params, unknowns, resids):
        self.callargs.append(params['X'].copy())


class FakeCompWithArrayFloatParams(Component):
    def __init__(self):
        super(FakeCompWithArrayFloatParams, self).__init__()
        self.add_param('X', val=np.array([0.]))
        self.add_param('y', val=0.0)
        self.add_param('Z', val=np.array([0.]))
        self.add_output('f_xy', val=0.0)
        self.callargs = []

    def solve_nonlinear(self, params, unknowns, resids):
        self.callargs.append((params['X'].copy(), params['y'].copy(), params['Z'].copy()))


class TestAllDOEDrivers(unittest.TestCase):

    def assertExpectedDoe(self, expected, doe):
        for i, sample in enumerate(doe):
            for j, v in enumerate(sample):
                val1 = expected[i][j]
                if isinstance(val1, list):
                    val1 = np.array(val1)
                else:
                    val1 = np.array([val1])
                if isinstance(v, list):
                    v = np.array(v)
                else:
                    v = np.array([v])
                assert_rel_error(self, val1, v, 1e-6)

    def runTestProblem1(self, driver):
        prob = Problem()
        root = prob.root = Group()

        comp = FakeCompWithArrayParam()
        root.add('comp', comp, promotes=['*'])
        root.add('p1', IndepVarComp('X', np.array([50., 50., 50.])), promotes=['*'])

        prob.driver = driver
        prob.driver.add_desvar('X',
                               lower=np.array([-50.0, -50.0, -50.0]),
                               upper=np.array([50.0, 50.0, 50.0]))
        prob.setup(check=False)
        prob.run()

        return comp.callargs

    def test_array_desvar(self):
        samples = self.runTestProblem1(LatinHypercubeDriver(num_samples=4, seed=SEED))
        #self.assertExpectedDoe(LHS1, samples)
        samples = self.runTestProblem1(FullFactorialDriver(num_levels=2))
        self.assertExpectedDoe(FACT1, samples)
        samples = self.runTestProblem1(UniformDriver(num_samples=4, seed=SEED))
        self.assertExpectedDoe(UNIFORM1, samples)

    def runTestProblem1_idx(self, driver):
        prob = Problem()
        root = prob.root = Group()

        comp = FakeCompWithArrayParam()
        root.add('comp', comp, promotes=['*'])
        root.add('p1', IndepVarComp('X', np.array([50., 50., 50.])), promotes=['*'])

        prob.driver = driver
        prob.driver.add_desvar('X',
                               lower=np.array([-50.0, -50.0]),
                               upper=np.array([50.0, 50.0]), indices=[1, 2])
        prob.setup(check=False)
        prob.run()

        return comp.callargs

    def test_array_desvar_idx(self):
        samples = self.runTestProblem1_idx(FullFactorialDriver(num_levels=2))
        self.assertExpectedDoe(FACT1[4:], samples)
        samples = self.runTestProblem1_idx(UniformDriver(num_samples=4, seed=SEED))
        self.assertAlmostEqual(samples[0][1], UNIFORM1[0][0])
        self.assertAlmostEqual(samples[0][2], UNIFORM1[0][1])
        self.assertAlmostEqual(samples[1][1], UNIFORM1[0][2])
        self.assertAlmostEqual(samples[1][2], UNIFORM1[1][0])
        self.assertAlmostEqual(samples[2][1], UNIFORM1[1][1])
        self.assertAlmostEqual(samples[2][2], UNIFORM1[1][2])

    def runTestProblem2(self, driver):
        prob = Problem()
        root = prob.root = Group()

        comp = FakeCompWithArrayFloatParams()
        root.add('comp', comp, promotes=['*'])
        root.add('p1', IndepVarComp('X', np.array([50.])), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('p3', IndepVarComp('Z', np.array([50.])), promotes=['*'])

        prob.driver = driver
        prob.driver.add_desvar('X',
                               lower=np.array([-50.0]),
                               upper=np.array([50.0]))
        prob.driver.add_desvar('y', lower=-50.0, upper=50)
        prob.driver.add_desvar('Z',
                               lower=np.array([-50.0]),
                               upper=np.array([50.0]))
        prob.setup(check=False)
        prob.run()

        return comp.callargs

    def test_mixed_array_float_desvar(self):
        samples = self.runTestProblem2(LatinHypercubeDriver(num_samples=4, seed=SEED))
        #self.assertExpectedDoe(LHS2, samples)
        samples = self.runTestProblem2(FullFactorialDriver(num_levels=2))
        self.assertExpectedDoe(FACT2, samples)
        samples = self.runTestProblem2(UniformDriver(num_samples=4, seed=SEED))
        self.assertExpectedDoe(UNIFORM2, samples)

    def runTestProblem2_scalar_bounds(self, driver):
        prob = Problem()
        root = prob.root = Group()

        comp = FakeCompWithArrayFloatParams()
        root.add('comp', comp, promotes=['*'])
        root.add('p1', IndepVarComp('X', np.array([50.])), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('p3', IndepVarComp('Z', np.array([50.])), promotes=['*'])

        prob.driver = driver
        prob.driver.add_desvar('X',
                               lower=-50.0,
                               upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50)
        prob.driver.add_desvar('Z',
                               lower=-50.0,
                               upper=50.0)
        prob.setup(check=False)
        prob.run()

        return comp.callargs

    def test_mixed_array_float_desvar_scalar_bounds(self):
        samples = self.runTestProblem2_scalar_bounds(LatinHypercubeDriver(num_samples=4, seed=SEED))
        #self.assertExpectedDoe(LHS2, samples)
        samples = self.runTestProblem2_scalar_bounds(FullFactorialDriver(num_levels=2))
        self.assertExpectedDoe(FACT2, samples)
        samples = self.runTestProblem2_scalar_bounds(UniformDriver(num_samples=4, seed=SEED))
        self.assertExpectedDoe(UNIFORM2, samples)
if __name__ == "__main__":
    unittest.main()
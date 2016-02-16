""" Testing driver LatinHypercubeDriver."""

import unittest
from random import seed
from types import GeneratorType

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, Component
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.util import assert_rel_error

from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver, OptimizedLatinHypercubeDriver
from openmdao.drivers.latinhypercube_driver import _is_latin_hypercube, _rand_latin_hypercube, _mmlhs, _LHC_Individual


class TestLatinHypercubeDriver(unittest.TestCase):

    def setUp(self):
        self.seed()
        self.hypercube_sizes = ((3, 1), (5, 5), (20, 8))

    def seed(self):
        # seedval = None
        self.seedval = 1
        seed(self.seedval)
        np.random.seed(self.seedval)

    def test_rand_latin_hypercube(self):
        for n, k in self.hypercube_sizes:
            test_lhc = _rand_latin_hypercube(n, k)

            self.assertTrue(_is_latin_hypercube(test_lhc))

    def _test_mmlhs_latin(self, n, k):
        p = 1
        population = 3
        generations = 6

        test_lhc = _rand_latin_hypercube(n, k)
        best_lhc = _LHC_Individual(test_lhc, 1, p)
        mmphi_initial = best_lhc.mmphi()
        for q in (1, 2, 5, 10, 20, 50, 100):
            lhc_start = _LHC_Individual(test_lhc, q, p)
            lhc_opt = _mmlhs(lhc_start, population, generations)
            if lhc_opt.mmphi() < best_lhc.mmphi():
                best_lhc = lhc_opt

        self.assertTrue(
                best_lhc.mmphi() < mmphi_initial,
                "'_mmlhs' didn't yield lower phi. Seed was {}".format(self.seedval))

    def test_mmlhs_latin(self):
        for n, k in self.hypercube_sizes:
            self._test_mmlhs_latin(n, k)

    def test_algorithm_coverage_lhc(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = LatinHypercubeDriver(100)

        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xDict = []
        yDict = []
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            x, y = dict(inputLine).values()
            xDict.append(np.floor(x))
            yDict.append(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == 100,
                "Incorrect number of runs generated.")

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xDict + yDict:
            if value < (-50) or value > 49:
                valuesInRange = False
        self.assertTrue(
                valuesInRange,
                "One of the input values was outside the given range.")

        # Assert a single input in each interval [n,n+1] for n = [-50,49]
        self.assertTrue(
                len(xDict) == 100,
                "One of the intervals wasn't covered.")
        self.assertTrue(
                len(yDict) == 100,
                "One of the intervals wasn't covered.")

    def test_algorithm_coverage_olhc(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = OptimizedLatinHypercubeDriver(100, population=5)
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xDict = []
        yDict = []
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            x, y = dict(inputLine).values()
            xDict.append(np.floor(x))
            yDict.append(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == 100,
                "Incorrect number of runs generated.")

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xDict + yDict:
            if value < (-50) or value > 49:
                valuesInRange = False
        self.assertTrue(
                valuesInRange,
                "One of the input values was outside the given range.")

        # Assert a single input in each interval [n,n+1] for n = [-50,49]
        self.assertTrue(
                len(xDict) == 100,
                "One of the intervals wasn't covered.")
        self.assertTrue(
                len(yDict) == 100,
                "One of the intervals wasn't covered.")

    '''
    def test_seed_works(self):


    def test_generate_numpydocstring(self):
        prob = Problem()
        prob.root = SellarStateConnection()
        prob.driver = ScipyOptimizer()

        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0]), upper=np.array([10.0]),
                              indices=[0])
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        test_string = prob.driver.generate_docstring()
        original_string = '    """\n\n    Options\n    -------\n    options[\'disp\'] :  bool(False)\n        Set to False to prevent printing of Scipy convergence messages\n    options[\'maxiter\'] :  int(200)\n        Maximum number of iterations.\n    options[\'optimizer\'] :  str(\'SLSQP\')\n        Name of optimizer to use\n    options[\'tol\'] :  float(1e-08)\n        Tolerance for termination. For detailed control, use solver-specific options.\n\n    """\n'
        self.assertEqual(original_string, test_string)
    '''

class ParaboloidArray(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(ParaboloidArray, self).__init__()

        self.add_param('X', val=np.array([0., 0.]))
        self.add_output('f_xy', val=0.0)

        self._history = []

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """

        x = params['X'][0]
        y = params['X'][1]
        self._history.append((x, y))
        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0


class TestLatinHypercubeDriverArray(unittest.TestCase):

    def test_rand_latin_hypercube(self):
        top = Problem()
        root = top.root = Group()

        root.add('p1', IndepVarComp('X', np.array([50., 50.])), promotes=['*'])

        root.add('comp', ParaboloidArray(), promotes=['*'])

        top.driver = OptimizedLatinHypercubeDriver(num_samples=4, seed=0, population=20,
                                                   generations=4, norm_method=2)
        top.driver.add_desvar('X', lower=np.array([-50., -50.]), upper=np.array([50., 50.]))

        top.driver.add_objective('f_xy')

        top.setup(check=False)
        top.run()

        results = top.root.comp._history
        #assert_rel_error(self, results[0][0], -11.279662, 1e-4)
        #assert_rel_error(self, results[0][1], -32.120265, 1e-4)
        #assert_rel_error(self, results[1][0], 40.069084, 1e-4)
        #assert_rel_error(self, results[1][1], -11.377920, 1e-4)
        #assert_rel_error(self, results[2][0], 10.5913699, 1e-4)
        #assert_rel_error(self, results[2][1], 41.147352826, 1e-4)
        #assert_rel_error(self, results[3][0], -39.06031971, 1e-4)
        #assert_rel_error(self, results[3][1], 22.29432501, 1e-4)

if __name__ == "__main__":
    unittest.main()

""" Testing driver LatinHypercubeDriver."""

from pprint import pformat
import unittest
from random import randint
from types import GeneratorType

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ScipyOptimizer, ExecComp
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivatives, SellarStateConnection
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error

from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver, OptimizedLatinHypercubeDriver
from openmdao.drivers.latinhypercube_driver import _is_latin_hypercube, _rand_latin_hypercube, _mmlhs, _LHC_Individual

class TestLatinHypercubeDriver(unittest.TestCase):

    def test_rand_latin_hypercube(self):

        n = randint(1,100)
        k = randint(1,20)

        test_lhc = _rand_latin_hypercube(n, k)

        self.assertTrue(_is_latin_hypercube(test_lhc))

    def test_mmlhs(self):

        n = randint(1,100)
        k = randint(1,20)
        p = 1
        population = 30
        generations = 6

        test_lhc = _rand_latin_hypercube(n, k)
        best_lhc = _LHC_Individual(test_lhc, 1, p)
        mmphi_initial = best_lhc.mmphi()
        for q in [1,2,5,10,20,50,100]:
            lhc_start = _LHC_Individual(test_lhc, q, p)
            lhc_opt = _mmlhs(lhc_start, population, generations)
            if lhc_opt.mmphi() < best_lhc.mmphi():
                best_lhc = lhc_opt

        self.assertTrue(
                best_lhc.mmphi() < mmphi_initial,
                "'_mmlhs' didn't yield lower phi.")

    def test_algorithm_coverage_lhc(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = LatinHypercubeDriver(100)
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)
        
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xDict = set()
        yDict = set()
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            [x, y] = dict(inputLine).values()
            xDict.add(np.floor(x))
            yDict.add(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == 100,
                "Incorrect number of runs generated.")

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xDict | yDict:
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

        prob.driver = OptimizedLatinHypercubeDriver(100)
        prob.driver.add_desvar('x', low=-50.0, high=50.0)
        prob.driver.add_desvar('y', low=-50.0, high=50.0)
        
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xDict = set()
        yDict = set()
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            [x, y] = dict(inputLine).values()
            xDict.add(np.floor(x))
            yDict.add(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == 100,
                "Incorrect number of runs generated.")

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xDict | yDict:
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

        prob.driver.add_desvar('z', low=np.array([-10.0]), high=np.array([10.0]),
                              indices=[0])
        prob.driver.add_desvar('x', low=0.0, high=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)
        prob.driver.options['disp'] = False

        test_string = prob.driver.generate_docstring()
        original_string = '    """\n\n    Options\n    -------\n    options[\'disp\'] :  bool(False)\n        Set to False to prevent printing of Scipy convergence messages\n    options[\'maxiter\'] :  int(200)\n        Maximum number of iterations.\n    options[\'optimizer\'] :  str(\'SLSQP\')\n        Name of optimizer to use\n    options[\'tol\'] :  float(1e-08)\n        Tolerance for termination. For detailed control, use solver-specific options.\n\n    """\n'
        self.assertEqual(original_string, test_string)
    '''

if __name__ == "__main__":
    unittest.main()

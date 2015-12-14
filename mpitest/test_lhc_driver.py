""" Testing driver LatinHypercubeDriver."""

import os
import unittest
from random import seed
from types import GeneratorType

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

from openmdao.test.util import assert_rel_error

from openmdao.test.paraboloid import Paraboloid

from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver, OptimizedLatinHypercubeDriver
from openmdao.drivers.latinhypercube_driver import _is_latin_hypercube, _rand_latin_hypercube, _mmlhs, _LHC_Individual


class TestLatinHypercubeDriver(MPITestCase):

    N_PROCS = 4

    def setUp(self):
        self.seed()
        self.hypercube_sizes = ((3, 1), (5, 5), (20, 8))

    def seed(self):
        # seedval = None
        self.seedval = 1
        seed(self.seedval)
        np.random.seed(self.seedval)

    def test_algorithm_coverage_lhc(self):

        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = LatinHypercubeDriver(100, num_par_doe=self.N_PROCS)

        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        prob.run()

        if MPI:
            runList = prob.driver._distrib_build_runlist()
            expected_runs = 25
        else:
            runList = prob.driver._build_runlist()
            expected_runs = 100

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xSet = set()
        ySet = set()
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            x, y = dict(inputLine).values()
            xSet.add(np.floor(x))
            ySet.add(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == expected_runs,
                "Incorrect number of runs generated. expected %d but got %d" %
                        (expected_runs, countRuns))

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xSet | ySet:
            if value < (-50) or value > 49:
                valuesInRange = False
        self.assertTrue(
                valuesInRange,
                "One of the input values was outside the given range.")

        # Assert a single input in each interval [n,n+1] for n = [-50,49]
        self.assertTrue(
                len(xSet) == expected_runs,
                "One of the intervals wasn't covered.")
        self.assertTrue(
                len(ySet) == expected_runs,
                "One of the intervals wasn't covered.")

    def test_algorithm_coverage_olhc(self):

        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = OptimizedLatinHypercubeDriver(100, population=5,
                                                    num_par_doe=self.N_PROCS)
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        prob.run()

        if MPI:
            runList = prob.driver._distrib_build_runlist()
            expected_runs = 25
        else:
            runList = prob.driver._build_runlist()
            expected_runs = 100

        # Ensure generated run list is a generator
        self.assertTrue(
                (type(runList) == GeneratorType),
                "_build_runlist did not return a generator.")

        # Add run list to dictionaries
        xSet = set()
        ySet = set()
        countRuns = 0
        for inputLine in runList:
            countRuns += 1
            x, y = dict(inputLine).values()
            xSet.add(np.floor(x))
            ySet.add(np.floor(y))

        # Assert we had the correct number of runs
        self.assertTrue(
                countRuns == expected_runs,
                "Incorrect number of runs generated. expected %d but got %d" %
                        (expected_runs, countRuns))

        # Assert all input values in range [-50,50]
        valuesInRange = True
        for value in xSet | ySet:
            if value < (-50) or value > 49:
                valuesInRange = False
        self.assertTrue(
                valuesInRange,
                "One of the input values was outside the given range.")

        # Assert a single input in each interval [n,n+1] for n = [-50,49]
        self.assertTrue(
                len(xSet) == expected_runs,
                "One of the intervals wasn't covered.")
        self.assertTrue(
                len(ySet) == expected_runs,
                "One of the intervals wasn't covered.")


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

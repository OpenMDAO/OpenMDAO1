#!/usr/bin/env python
""" Running DOE experiments (in parallel)"""

from __future__ import print_function

import time
import numpy as np

from openmdao.api import IndepVarComp
from openmdao.api import Component, Group
from openmdao.api import Problem
from openmdao.api import FullFactorialDriver
from openmdao.api import InMemoryRecorder

from openmdao.core.mpi_wrap import MPI, debug
from openmdao.test.mpi_util import MPITestCase

if MPI: # pragma: no cover
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.api import BasicImpl as impl

class Mult(Component):
    def __init__(self):
        super(Mult, self).__init__()
        self.add_param('x', val=0.)
        self.add_param('c', val=0.)
        self.add_output('y', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            time.sleep((MPI.COMM_WORLD.rank+1.0)*0.1)
        unknowns['y'] = params['c']*params['x']


class ParallelDOETestCase(MPITestCase):

    N_PROCS = 4

    def test_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', Mult())

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

class LBParallelDOETestCase(MPITestCase):

    N_PROCS = 5

    def test_load_balanced_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', Mult())

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

class LBParallelDOETestCase6(MPITestCase):

    N_PROCS = 6

    def test_load_balanced_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', Mult())

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

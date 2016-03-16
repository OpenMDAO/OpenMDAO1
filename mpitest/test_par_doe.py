#!/usr/bin/env python
""" Running DOE experiments (in parallel)"""

from __future__ import print_function

import time
import traceback
import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem, \
                         FullFactorialDriver, InMemoryRecorder, AnalysisError

from openmdao.core.mpi_wrap import MPI, debug, MultiProcFailCheck
from openmdao.test.mpi_util import MPITestCase

if MPI: # pragma: no cover
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.api import BasicImpl as impl

class Mult(Component):
    def __init__(self, fails=(), critical=False):
        super(Mult, self).__init__()
        self.iter_count = 0
        self.fails = fails # case numbers to fail on
        self.critical = critical

        self.add_param('x', val=0.)
        self.add_param('c', val=0.)
        self.add_output('y', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            time.sleep((MPI.COMM_WORLD.rank+1.0)*0.1)
        unknowns['y'] = params['c']*params['x']

        try:
            if self.iter_count in self.fails:
                if self.critical:
                    raise RuntimeError("OMG, a critical error!")
                else:
                    raise AnalysisError("just an analysis error: %d" % MPI.COMM_WORLD.rank)
        finally:
            self.iter_count += 1


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

    def test_doe_fail_critical(self):
        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        if MPI:
            fail_rank = 1  # raise exception from this rank
        else:
            fail_rank = 0
            
        if self.comm.rank == fail_rank:
            root.add('mult', Mult(fails=[3], critical=True))
        else:
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

        try:
            problem.run()
        except Exception as err:
            with MultiProcFailCheck(self.comm):
                if self.comm.rank == fail_rank:
                    self.assertEqual(str(err), "OMG, a critical error!")
                else:
                    self.assertEqual(str(err),
                            "an exception was raised by another MPI process.")

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), 12)
        else:
            self.assertEqual(num_cases, 3)

    def test_doe_fail_analysis_error(self):
        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        fail_rank = 1  # raise exception from this rank
        if self.comm.rank == fail_rank:
            debug("fail rank!", self.comm.rank)
            root.add('mult', Mult(fails=[3, 4]))
        else:
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
            self.assertEqual(sum(lens), 25)
        else:
            self.assertEqual(num_cases, 25)

        nfails = 0
        for data in problem.driver.recorders[0].iters:
            if not data['success']:
                nfails += 1
                debug(data['msg'])

        debug("nfails:",nfails)
        if self.comm.rank == fail_rank:
            self.assertEqual(nfails, 2)
        else:
            self.assertEqual(nfails, 0)


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

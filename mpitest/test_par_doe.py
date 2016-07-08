#!/usr/bin/env python
""" Running DOE experiments (in parallel)"""

from __future__ import print_function

import sys
import time
import traceback
import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem, \
                         FullFactorialDriver, AnalysisError
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.core.mpi_wrap import MPI, debug, MultiProcFailCheck
from openmdao.test.mpi_util import MPITestCase

if MPI: # pragma: no cover
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.api import BasicImpl as impl

class ParallelDOETestCase(MPITestCase):

    N_PROCS = 4

    def test_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test("y=c*x"))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                             num_par_doe=self.N_PROCS)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_response(['indep_var.x', 'mult.y'])

        problem.setup(check=False)
        problem.run()

        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            self.assertEqual(responses['indep_var.x']*2.0,
                             responses['mult.y'])
            num_cases += 1

        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

    def test_doe_fail_hard(self):
        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        if MPI:
            fail_rank = 1  # raise exception from this rank
            npardoe = self.N_PROCS
        else:
            fail_rank = 0
            npardoe = 1

        root.add('mult', ExecComp4Test("y=c*x", fail_rank=fail_rank,
                 fails=[3], fail_hard=True))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                             num_par_doe=npardoe)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])

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
        else:
            if MPI:
                self.fail('exception expected')

        nsucc = 0
        nfail = 0
        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            if success:
                self.assertEqual(responses['indep_var.x']*2.0,
                                 responses['mult.y'])
                nsucc += 1
            else:
                nfail += 1

        self.assertEqual(nfail, 0) # hard errors aren't saved

        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), 12)
        else:
            self.assertTrue(num_cases < num_levels)

    def test_doe_fail_analysis_error(self):
        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        if MPI:
            fail_rank = 1  # raise exception from this rank
            npardoe = self.N_PROCS
        else:
            fail_rank = 0
            npardoe = 1

        root.add('mult', ExecComp4Test("y=c*x", fail_rank=fail_rank,
                  fails=[3,4]))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                             num_par_doe=npardoe)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])
        problem.driver.add_response('mult.case_rank')

        problem.setup(check=False)

        problem.run()

        fails_in_fail_rank = 0
        nfails = 0
        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            if success:
                self.assertEqual(responses['indep_var.x']*2.0,
                                 responses['mult.y'])
            else:
                nfails += 1
                if responses['mult.case_rank'] == fail_rank:
                    fails_in_fail_rank += 1

        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), 25)
            if self.comm.rank == 0:
                self.assertEqual(fails_in_fail_rank, 2)
                self.assertEqual(nfails, 2)
        else:
            self.assertEqual(num_cases, 25)
            self.assertEqual(nfails, 2)


class LBParallelDOETestCase(MPITestCase):

    N_PROCS = 5

    def test_load_balanced_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test("y=c*x"))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])

        problem.setup(check=False)
        problem.run()

        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            if success:
                self.assertEqual(responses['indep_var.x']*2.0,
                                 responses['mult.y'])

        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

    def test_load_balanced_doe_crit_fail(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        fail_rank = 1  # raise exception from this rank

        root.add('mult', ExecComp4Test("y=c*x", fail_rank=fail_rank,
                 fails=[1], fail_hard=True))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])

        problem.setup(check=False)
        problem.run()

        nsucc = 0
        nfail = 0
        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            if success:
                self.assertEqual(responses['indep_var.x']*2.0,
                                 responses['mult.y'])
                nsucc += 1
            else:
                nfail += 1

        # in load balanced mode, we can't really predict how many cases
        # will actually run before we terminate, so just check to see if
        # we at least have less than the full set we'd have if nothing
        # went wrong.
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertTrue(sum(lens) < num_levels,
                    "Cases run (%d) should be less than total cases (%d)" %
                    (sum(lens), num_levels))
        else:
            self.assertTrue(num_cases < num_levels)
            self.assertEqual(nfail, 0) # hard failure cases are not saved

    def test_load_balanced_doe_soft_fail(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        if MPI:
            fail_rank = 1  # raise exception from this rank
        else:
            fail_rank = 0

        fail_idxs = [3,4,5]
        root.add('mult', ExecComp4Test("y=c*x", fail_rank=fail_rank,
                 fails=fail_idxs))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])
        problem.driver.add_response('mult.case_rank')

        problem.setup(check=False)
        problem.run()

        nfails = 0
        num_cases = 0
        cases_in_fail_rank = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            if success:
                self.assertEqual(responses['indep_var.x']*2.0,
                                 responses['mult.y'])
            else:
                nfails += 1
            if responses['mult.case_rank'] == fail_rank:
                cases_in_fail_rank += 1

        if MPI and self.comm.rank > 0:
            self.assertEqual(num_cases, 0)
        else:
            self.assertEqual(num_cases, num_levels)

        if self.comm.rank == 0:
            # FIXME: for now, all cases get sent back to the master process (0),
            # even when recorders are parallel.

            # there's a chance that the fail rank didn't get enough
            # cases to actually fail 3 times, so we need to check
            # how many cases it actually got.

            if cases_in_fail_rank > 5:
                self.assertEqual(nfails, 3)
            elif cases_in_fail_rank > 4:
                self.assertEqual(nfails, 2)
            elif cases_in_fail_rank > 3:
                self.assertEqual(nfails, 1)
            else:
                self.assertEqual(nfails, 0)
        else:
            self.assertEqual(nfails, 0)

class LBParallelDOETestCase6(MPITestCase):

    N_PROCS = 6

    def test_load_balanced_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test("y=c*x"))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=self.N_PROCS,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')
        problem.driver.add_response(['indep_var.x', 'mult.y'])

        problem.setup(check=False)
        problem.run()

        num_cases = 0
        for responses, success, msg in problem.driver.get_responses():
            num_cases += 1
            self.assertEqual(responses['indep_var.x']*2.0,
                             responses['mult.y'])

        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_levels)
        else:
            self.assertEqual(num_cases, num_levels)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

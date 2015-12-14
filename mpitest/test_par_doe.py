#!/usr/bin/env python
""" Running DOE experiments (in parallel)"""

from __future__ import print_function

import time
import numpy as np

from openmdao.api import IndepVarComp
from openmdao.api import Component, Group
from openmdao.api import Problem
from openmdao.api import UniformDriver
from openmdao.api import InMemoryRecorder
from openmdao.api import NLGaussSeidel

from openmdao.core.mpi_wrap import MPI, MultiProcFailCheck
from openmdao.test.mpi_util import MPITestCase

if MPI: # pragma: no cover
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.api import BasicImpl as impl

class DUT(Component):
    def __init__(self):
        super(DUT, self).__init__()
        self.add_param('x', val=0.)
        self.add_param('c', val=0.)
        self.add_output('y', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much.  Just multiply by 3"""
        time.sleep(.5)
        unknowns['y'] = params['c']*params['x']

class ParallelDOETestCase(MPITestCase):

    N_PROCS = 4

    def test_doe(self):

        problem = Problem(impl=impl)
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=7.0))
        root.add('const', IndepVarComp('c', val=3.0))
        root.add('dut', DUT())

        root.connect('indep_var.x', 'dut.x')
        root.connect('const.c', 'dut.c')

        num_samples = 10
        problem.driver = UniformDriver(num_samples=num_samples,
                                       num_par_doe=self.N_PROCS)
        problem.driver.add_desvar('indep_var.x', low=4410.0,  high=4450.0)
        problem.driver.add_objective('dut.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*3.0,
                             data['unknowns']['dut.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        if MPI:
            lens = problem.comm.allgather(num_cases)
            self.assertEqual(sum(lens), num_samples)
        else:
            self.assertEqual(num_cases, num_samples)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

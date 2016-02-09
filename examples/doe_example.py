#!/usr/bin/env python
""" Running DOE experiments (in parallel)"""

from __future__ import print_function

import numpy as np

from openmdao.api import IndepVarComp
from openmdao.api import Component, Group
from openmdao.api import Problem
from openmdao.api import UniformDriver
from openmdao.api import DumpRecorder
from openmdao.api import NLGaussSeidel

from openmdao.core.mpi_wrap import MPI

class DUT(Component):
    def __init__(self):
        super(DUT, self).__init__()
        self.add_param('x', val=0.)
        self.add_param('c', val=0.)
        self.add_output('y', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much.  Just multiply x by c"""
        sum([j*j for j in xrange(10000000)])    # dummy delay (busy loop)
        unknowns['y'] = params['c']*params['x']

if __name__ == "__main__":

    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl
    else:
        # if you didn't use `mpirun`, then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    problem = Problem(impl=impl)
    root = problem.root = Group()
    root.add('indep_var', IndepVarComp('x', val=7.0))
    root.add('const', IndepVarComp('c', val=3.0, pass_by_obj=False))
    root.add('dut', DUT())

    root.connect('indep_var.x', 'dut.x')
    root.connect('const.c', 'dut.c')

    problem.driver = UniformDriver(num_samples = 10, num_par_doe=5)
    problem.driver.add_desvar('indep_var.x', low=4410.0,  high=4450.0)
    problem.driver.add_objective('dut.y')

    recorder = DumpRecorder()
    problem.driver.add_recorder(recorder)

    problem.setup()
    problem.run()

    problem['dut.y']

    problem.cleanup()

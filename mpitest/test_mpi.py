import sys
import unittest
import time

import numpy as np

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.component import Component
from openmdao.core.mpi_wrap import MPI, MultiProcFailCheck

from openmdao.components.indep_var_comp import IndepVarComp

from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core import BasicImpl as impl

from openmdao.test.util import assert_rel_error


class ABCDArrayComp(Component):

    def __init__(self, arr_size=9, delay=0.01):
        super(ABCDArrayComp, self).__init__()
        self.add_param('a', np.ones(arr_size, float))
        self.add_param('b', np.ones(arr_size, float))
        self.add_param('in_string', '')
        self.add_param('in_list', [])

        self.add_output('c', np.ones(arr_size, float))
        self.add_output('d', np.ones(arr_size, float))
        self.add_output('out_string', '')
        self.add_output('out_list', [])

        self.delay = delay

    def solve_nonlinear(self, params, unknowns, resids):
        time.sleep(self.delay)

        unknowns['c'] = params['a'] + params['b']
        unknowns['d'] = params['a'] - params['b']

        unknowns['out_string'] = params['in_string'] + '_' + self.name
        unknowns['out_list']   = params['in_list'] + [1.5]

class MPITests1(MPITestCase):

    N_PROCS = 1

    def test_simple(self):
        prob = Problem(Group(), impl=impl)

        size = 5
        A1 = prob.root.add('A1', IndepVarComp('a', np.zeros(size, float)))
        B1 = prob.root.add('B1', IndepVarComp('b', np.zeros(size, float)))
        B2 = prob.root.add('B2', IndepVarComp('b', np.zeros(size, float)))
        S1 = prob.root.add('S1', IndepVarComp('s', ''))
        L1 = prob.root.add('L1', IndepVarComp('l', []))

        C1 = prob.root.add('C1', ABCDArrayComp(size))
        C2 = prob.root.add('C2', ABCDArrayComp(size))

        prob.root.connect('A1.a', 'C1.a')
        prob.root.connect('B1.b', 'C1.b')
        # prob.root.connect('S1:s', 'C1.in_string')
        # prob.root.connect('L1:l', 'C1.in_list')

        prob.root.connect('C1.c', 'C2.a')
        prob.root.connect('B2.b', 'C2.b')
        # prob.root.connect('C1.out_string', 'C2.in_string')
        # prob.root.connect('C1.out_list',   'C2.in_list')

        prob.setup(check=False)

        prob['A1.a'] = np.ones(size, float) * 3.0
        prob['B1.b'] = np.ones(size, float) * 7.0
        prob['B2.b'] = np.ones(size, float) * 5.0

        prob.run()

        self.assertTrue(all(prob['C2.a']==np.ones(size, float)*10.))
        self.assertTrue(all(prob['C2.b']==np.ones(size, float)*5.))
        self.assertTrue(all(prob['C2.c']==np.ones(size, float)*15.))
        self.assertTrue(all(prob['C2.d']==np.ones(size, float)*5.))

        # TODO: can't do MPI pass_by_object yet
        # self.assertTrue(prob['C2.out_string']=='_C1_C2')
        # self.assertTrue(prob['C2.out_list']==[1.5, 1.5])

class MPITests2(MPITestCase):

    N_PROCS = 2

    def test_too_many_procs(self):
        prob = Problem(Group(), impl=impl)

        size = 5
        A1 = prob.root.add('A1', IndepVarComp('a', np.zeros(size, float)))
        C1 = prob.root.add('C1', ABCDArrayComp(size))

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "This problem was given 2 MPI processes, "
                             "but it requires between 1 and 1.")
        else:
            if MPI:
                self.fail("Exception expected")

    def test_parallel_fan_in(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.setup(check=False)
        prob.run()

        if not MPI or self.comm.rank == 0:
            self.assertTrue(all(prob.root.C1.params['a']==np.ones(size, float)*1.0))
            self.assertTrue(all(prob.root.C1.params['b']==np.ones(size, float)*2.0))
            self.assertTrue(all(prob['C1.c']==np.ones(size, float)*3.0))
            self.assertTrue(all(prob['C1.d']==np.ones(size, float)*-1.0))
            # TODO: not handling non-flattenable vars yet

    def test_parallel_diamond(self):
        size = 3
        prob = Problem(Group(), impl=impl)
        root = prob.root
        root.add('P1', IndepVarComp('x', np.ones(size, float) * 1.1))
        G1 = root.add('G1', ParallelGroup())
        G1.add('C1', ABCDArrayComp(size))
        G1.add('C2', ABCDArrayComp(size))
        root.add('C3', ABCDArrayComp(size))

        root.connect('P1.x', 'G1.C1.a')
        root.connect('P1.x', 'G1.C2.b')
        root.connect('G1.C1.c', 'C3.a')
        root.connect('G1.C2.d', 'C3.b')

        prob.setup(check=False)
        prob.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, prob.root.G1.C1.unknowns['c'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.G1.C1.unknowns['d'],
                             np.ones(size)*.1, 1.e-10)
            assert_rel_error(self, prob.root.C3.params['a'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.C3.params['b'],
                             np.ones(size)*-.1, 1.e-10)

        if not MPI or self.comm.rank == 1:
            assert_rel_error(self, prob.root.G1.C2.unknowns['c'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.G1.C2.unknowns['d'],
                             np.ones(size)*-.1, 1.e-10)

    def test_wrong_impl(self):
        if MPI:
            try:
                prob = Problem(Group())
            except Exception as err:
                self.assertEqual(str(err),
                   "To run under MPI, the impl for a Problem must be PetscImpl.")
            else:
                self.fail("Exception expected")

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

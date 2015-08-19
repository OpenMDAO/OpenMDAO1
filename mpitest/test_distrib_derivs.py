""" Test out some crucial linear GS tests in parallel with distributed comps."""

from __future__ import print_function

import sys
import numpy

from openmdao.core.mpi_wrap import MPI, MultiProcFailCheck
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.component import Component
from openmdao.components.param_comp import ParamComp
from openmdao.components.exec_comp import ExecComp
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.simple_comps import FanInGrouped, FanOutGrouped
from openmdao.test.util import assert_rel_error
from openmdao.util.array_util import evenly_distrib_idxs

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    rank = MPI.COMM_WORLD.rank
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    rank = 0

class DistribExecComp(ExecComp):
    """An ExecComp that can only use variables x and y.  It uses 2 procs and
    takes input var slices and has output var slices as well.
    """
    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size

    def setup_distrib_idxs(self):
        """ component declares the local sizes and sets initial values
        for all distributed inputs and outputs. Returns a dict of
        index arrays keyed to variable names.
        """
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        print(rank, 'src_indices:',start,'to',end)
        self.set_var_indices('x', val=numpy.ones(sizes[rank], float),
                             src_indices=numpy.arange(start, end, dtype=int))
        self.set_var_indices('y', val=numpy.ones(sizes[rank], float),
                             src_indices=numpy.arange(start, end, dtype=int))

    def get_req_cpus(self):
        return (2, 2)

class MPITests1(MPITestCase):

    N_PROCS = 2

    def test_two_simple(self):
        size = 3
        group = Group()
        group.add('P', ParamComp('x', numpy.ones(size)))
        group.add('C1', DistribExecComp(['y=2.0*x'], arr_size=size,
                                           x=numpy.zeros(size),
                                           y=numpy.zeros(size)))
        group.add('C2', ExecComp(['z=3.0*y'], y=numpy.zeros(size),
                                           z=numpy.zeros(size)))

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.connect('P.x', 'C1.x')
        prob.root.connect('C1.y', 'C2.y')

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='fwd', return_format='dict')
        print("J:",J)
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

        J = prob.calc_gradient(['P.x'], ['C2.z'], mode='rev', return_format='dict')
        print("rev J:",J)
        assert_rel_error(self, J['C2.z']['P.x'], numpy.eye(size)*6.0, 1e-6)

    def test_fan_out_grouped(self):
        size = 3
        prob = Problem(impl=impl)
        prob.root = root = Group()
        root.add('P', ParamComp('x', numpy.ones(size, dtype=float)))
        root.add('C1', DistribExecComp(['y=3.0*x'], arr_size=size,
                                   x=numpy.zeros(size, dtype=float),
                                   y=numpy.zeros(size, dtype=float)))
        sub = root.add('sub', ParallelGroup())
        sub.add('C2', ExecComp('y=1.5*x',
                               x=numpy.zeros(size),
                               y=numpy.zeros(size)))
        sub.add('C3', ExecComp(['y=5.0*x'],
                                   x=numpy.zeros(size, dtype=float),
                                   y=numpy.zeros(size, dtype=float)))

        root.add('C2', ExecComp(['y=x'],
                                   x=numpy.zeros(size, dtype=float),
                                   y=numpy.zeros(size, dtype=float)))
        root.add('C3', ExecComp(['y=x'],
                                   x=numpy.zeros(size, dtype=float),
                                   y=numpy.zeros(size, dtype=float)))
        root.connect('sub.C2.y', 'C2.x')
        root.connect('sub.C3.y', 'C3.x')

        root.connect("C1.y", "sub.C2.x")
        root.connect("C1.y", "sub.C3.x")
        root.connect("P.x", "C1.x")

        root.ln_solver = LinearGaussSeidel()
        root.sub.ln_solver = LinearGaussSeidel()
        prob.setup(check=False)

        prob.run()

        param_list = ['P.x']
        unknown_list = ['C2.y', "C3.y"]

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        print("J:",J)
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        print("rev J:",J)
        assert_rel_error(self, J['C2.y']['P.x'], numpy.eye(size)*4.5, 1e-6)
        assert_rel_error(self, J['C3.y']['P.x'], numpy.eye(size)*15.0, 1e-6)
    #
    # def test_fan_in_grouped(self):
    #
    #     prob = Problem(impl=impl)
    #     prob.root = FanInGrouped()
    #     prob.root.ln_solver = LinearGaussSeidel()
    #     prob.root.sub.ln_solver = LinearGaussSeidel()
    #     prob.setup(check=False)
    #     prob.run()
    #
    #     param_list = ['p1.x1', 'p2.x2']
    #     unknown_list = ['comp3.y']
    #
    #     J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)
    #
    #     J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

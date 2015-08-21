""" Unit test for the Scipy GMRES linear solver. """
from __future__ import print_function

import unittest
import numpy as np

from openmdao.components.param_comp import ParamComp

from openmdao.core.group import Group
from openmdao.core.problem import Problem

from openmdao.solvers.petsc_ksp import PetscKSP
from openmdao.test.converge_diverge import ConvergeDivergePar, SingleDiamondPar
from openmdao.test.simple_comps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                       FanOutGrouped, FanInGrouped, ArrayComp2D
from openmdao.test.util import assert_rel_error

from openmdao.core.mpi_wrap import MPI, MultiProcFailCheck
from openmdao.test.mpi_util import MPITestCase
from openmdao.devtools.debug import debug

try:
    from mpi4py import MPI
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

class TestPetscKSP(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if impl is None:
            raise unittest.SkipTest("Can't run this test (even in serial) without mpi4py and petsc4py")

    def test_fan_out_grouped(self):

        prob = Problem(impl=impl)
        prob.root = FanOutGrouped()
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        param_list = ['p.x']
        #currently, you can't have vars of interest that are down in a parallel system
        #unknown_list = ['sub.comp2.y', "sub.comp3.y"]
        unknown_list = ['c2.y', "c3.y"]

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J[unknown_list[0]]['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J[unknown_list[1]]['p.x'][0][0], 15.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J[unknown_list[0]]['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J[unknown_list[1]]['p.x'][0][0], 15.0, 1e-6)

    def test_simple_deriv_xfer(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.setup(check=False)

        prob.root.comp3.dpmat[None]['x1'] = 7.
        prob.root.comp3.dpmat[None]['x2'] = 11.
        prob.root._transfer_data(mode='rev', deriv=True)

        if not MPI or self.comm.rank == 0:
            self.assertEqual(prob.root.sub.comp1.dumat[None]['y'], 7.)

        if not MPI or self.comm.rank == 1:
            self.assertEqual(prob.root.sub.comp2.dumat[None]['y'], 11.)

        prob.root.comp3.dpmat[None]['x1'] = 0.
        prob.root.comp3.dpmat[None]['x2'] = 0.
        self.assertEqual(prob.root.comp3.dpmat[None]['x1'], 0.)
        self.assertEqual(prob.root.comp3.dpmat[None]['x2'], 0.)

        prob.root._transfer_data(mode='fwd', deriv=True)

        self.assertEqual(prob.root.comp3.dpmat[None]['x1'], 7.)
        self.assertEqual(prob.root.comp3.dpmat[None]['x2'], 11.)

    def test_fan_in(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()
        prob.root.ln_solver = PetscKSP()

        prob.setup(check=False)

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
            assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
            assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_single_diamond(self):

        prob = Problem(impl=impl)
        prob.root = SingleDiamondPar()
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        param_list = ['p.x']
        unknown_list = ['comp4.y1', 'comp4.y2']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    def test_converge_diverge_groups(self):

        prob = Problem(impl=impl)
        prob.root = ConvergeDivergePar()
        prob.root.ln_solver = PetscKSP()

        prob.setup(check=False)
        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

        param_list = ['p.x']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

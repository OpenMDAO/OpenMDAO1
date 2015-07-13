""" Unit test for the Scipy GMRES linear solver. """

import unittest
import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.simplecomps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                      FanOutGrouped, \
                                      FanInGrouped, ArrayComp2D
from openmdao.test.testutil import assert_rel_error

from openmdao.core.mpiwrap import MPI, MultiProcFailCheck
from openmdao.test.mpiunittest import MPITestCase
from openmdao.devtools.debug import debug

if MPI:
    from openmdao.core.petscimpl import PetscImpl as impl
else:
    from openmdao.core.basicimpl import BasicImpl as impl

from openmdao.test.mpiunittest import MPITestCase

class TestScipyGMRES(MPITestCase):

    N_PROCS = 2

    #def test_fan_out_grouped(self):

        #top = Problem()
        #top.root = FanOutGrouped()
        #top.root.ln_solver = ScipyGMRES()
        #top.setup()
        #top.run()

        #param_list = ['p.x']
        #unknown_list = ['sub.comp2.y', "sub.comp3.y"]

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_simple_deriv_xfer(self):

        top = Problem(impl=impl)
        top.root = FanInGrouped()
        top.setup()

        top.root.comp3.dpmat[None]['x1'] = 7.
        top.root.comp3.dpmat[None]['x2'] = 11.
        top.root._transfer_data(mode='rev', deriv=True)

        if not MPI or self.comm.rank == 0:
            self.assertEqual(top.root.sub.comp1.dumat[None]['y'], 7.)

        if not MPI or self.comm.rank == 1:
            self.assertEqual(top.root.sub.comp2.dumat[None]['y'], 11.)

        top.root.comp3.dpmat[None]['x1'] = 0.
        top.root.comp3.dpmat[None]['x2'] = 0.
        self.assertEqual(top.root.comp3.dpmat[None]['x1'], 0.)
        self.assertEqual(top.root.comp3.dpmat[None]['x2'], 0.)

        top.root._transfer_data(mode='fwd', deriv=True)

        self.assertEqual(top.root.comp3.dpmat[None]['x1'], 7.)
        self.assertEqual(top.root.comp3.dpmat[None]['x2'], 11.)

    #def test_fan_in(self):

        #top = Problem(impl=impl)
        #top.root = FanInGrouped()
        #top.setup()

        #param_list = ['p1.x1', 'p2.x2']
        #unknown_list = ['comp3.y']

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #if not MPI or self.comm.rank == 0:
            #assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
            #assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #if not MPI or self.comm.rank == 0:
            #assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
            #assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    #def test_converge_diverge_groups(self):

        #top = Problem()
        #top.root = ConvergeDivergeGroups()
        #top.root.ln_solver = ScipyGMRES()
        #top.setup()
        #top.run()

        ## Make sure value is fine.
        #assert_rel_error(self, top['comp7.y1'], -102.7, 1e-6)

        #param_list = ['p.x']
        #unknown_list = ['comp7.y1']

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        #assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #def test_single_diamond(self):

        #top = Problem()
        #top.root = SingleDiamond()
        #top.root.ln_solver = ScipyGMRES()
        #top.setup()
        #top.run()

        #param_list = ['p.x']
        #unknown_list = ['comp4.y1', 'comp4.y2']

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        #assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        #assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    #def test_single_diamond_grouped(self):

        #top = Problem()
        #top.root = SingleDiamondGrouped()
        #top.root.ln_solver = ScipyGMRES()
        #top.setup()
        #top.run()

        #param_list = ['p.x']
        #unknown_list = ['comp4.y1', 'comp4.y2']

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        #assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        #assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        #J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        #assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        #assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)


if __name__ == '__main__':
    from openmdao.test.mpiunittest import mpirun_tests
    mpirun_tests()

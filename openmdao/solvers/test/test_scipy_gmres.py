""" Unit test for the Scipy GMRES linear solver. """

import unittest
from unittest import SkipTest

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.simplecomps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                      SimpleCompDerivJac, FanOutGrouped, \
                                      FanInGrouped, ArrayComp2D
from openmdao.test.testutil import assert_rel_error


class TestScipyGMRES(unittest.TestCase):

    def test_simple_matvec(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_array2D(self):
        group = Group()
        group.add('x_param', ParamComp('x', np.ones((2, 2))), promotes=['*'])
        group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = top.subsystem('mycomp')._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_jac(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivJac(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_fan_out(self):

        top = Problem()
        top.root = FanOut()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp2:y', "comp3:y"]

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp2:y']['p:x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p:x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp2:y']['p:x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p:x'][0][0], 15.0, 1e-6)

    def test_fan_out_grouped(self):

        top = Problem()
        top.root = FanOutGrouped()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['sub:comp2:y', "sub:comp3:y"]

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub:comp2:y']['p:x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub:comp3:y']['p:x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub:comp2:y']['p:x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub:comp3:y']['p:x'][0][0], 15.0, 1e-6)

    def test_fan_in(self):

        top = Problem()
        top.root = FanIn()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p1:x1', 'p2:x2']
        unknown_list = ['comp3:y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3:y']['p1:x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p2:x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3:y']['p1:x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p2:x2'][0][0], 35.0, 1e-6)

    def test_fan_in_grouped(self):

        top = Problem()
        top.root = FanInGrouped()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p1:x1', 'p2:x2']
        unknown_list = ['comp3:y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3:y']['p1:x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p2:x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3:y']['p1:x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3:y']['p2:x2'][0][0], 35.0, 1e-6)

    def test_converge_diverge(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp7:y1']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

    def test_converge_diverge_groups(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp7:y1']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

    def test_single_diamond(self):

        top = Problem()
        top.root = SingleDiamond()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp4:y1', 'comp4:y2']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)

    def test_single_diamond_grouped(self):

        top = Problem()
        top.root = SingleDiamondGrouped()
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp4:y1', 'comp4:y2']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)


if __name__ == "__main__":
    unittest.main()

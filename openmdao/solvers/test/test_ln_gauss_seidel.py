""" Unit test for the Gauss Seidel linear solver. """

import unittest
from unittest import SkipTest

import numpy as np

from openmdao.components.execcomp import ExecComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.simplecomps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                      FanOutGrouped, \
                                      FanInGrouped, ArrayComp2D
from openmdao.test.testutil import assert_rel_error


class TestLinearGaussSeidel(unittest.TestCase):

    def test_simple_matvec(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = Group()
        top.root.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        top.root.add('sub', group, promotes=['*'])

        top.root.ln_solver = LinearGaussSeidel()
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
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = top.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_two_simple(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0))
        group.add('comp1', ExecComp(['y=2.0*x']))
        group.add('comp2', ExecComp(['z=3.0*y']))

        top = Problem()
        top.root = group
        top.root.ln_solver = LinearGaussSeidel()
        top.root.connect('x_param.x', 'comp1.x')
        top.root.connect('comp1.y', 'comp2.y')

        top.setup()
        top.run()

        J = top.calc_gradient(['x_param.x'], ['comp2.z'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp2.z']['x_param.x'][0][0], 6.0, 1e-6)

        J = top.calc_gradient(['x_param.x'], ['comp2.z'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp2.z']['x_param.x'][0][0], 6.0, 1e-6)

    def test_fan_out(self):

        top = Problem()
        top.root = FanOut()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p.x']
        unknown_list = ['comp2.y', "comp3.y"]

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_out_grouped(self):

        top = Problem()
        top.root = FanOutGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p.x']
        unknown_list = ['sub.comp2.y', "sub.comp3.y"]

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_in(self):

        top = Problem()
        top.root = FanIn()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_grouped(self):

        top = Problem()
        top.root = FanInGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_converge_diverge(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p.x']
        unknown_list = ['comp7.y1']

        top.run()

        # Make sure value is fine.
        assert_rel_error(self, top['comp7.y1'], -102.7, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    def test_converge_diverge_groups(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        # Make sure value is fine.
        assert_rel_error(self, top['comp7.y1'], -102.7, 1e-6)

        param_list = ['p.x']
        unknown_list = ['comp7.y1']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    def test_single_diamond(self):

        top = Problem()
        top.root = SingleDiamond()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p.x']
        unknown_list = ['comp4.y1', 'comp4.y2']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    def test_single_diamond_grouped(self):

        top = Problem()
        top.root = SingleDiamondGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.setup()
        top.run()

        param_list = ['p.x']
        unknown_list = ['comp4.y1', 'comp4.y2']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)


if __name__ == "__main__":
    unittest.main()

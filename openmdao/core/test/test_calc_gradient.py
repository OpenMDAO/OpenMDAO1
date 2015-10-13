""" Unit tests for the calc_gradient method on Problem. """

from __future__ import print_function

import unittest
import numpy as np
from six import text_type, PY3

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.components.exec_comp import ExecComp
from openmdao.test.simple_comps import RosenSuzuki, FanIn


if PY3:
    def py3fix(s):
        return s.replace('<type', '<class')
else:
    def py3fix(s):
        return s


class TestCalcGradient(unittest.TestCase):

    def test_calc_gradient_interface_errors(self):

        root = Group()
        prob = Problem(root=root)
        root.add('comp', ExecComp('y=x*2.0'))

        try:
            prob.calc_gradient(['comp.x'], ['comp.y'], mode='junk')
        except Exception as error:
            msg = "mode must be 'auto', 'fwd', 'rev', or 'fd'"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

        try:
            prob.calc_gradient(['comp.x'], ['comp.y'], return_format='junk')
        except Exception as error:
            msg = "return_format must be 'array' or 'dict'"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_calc_gradient(self):
        root = Group()
        root.add('parm', IndepVarComp('x', np.array([1., 1., 1., 1.])))
        root.add('comp', RosenSuzuki())

        root.connect('parm.x', 'comp.x')

        prob = Problem(root)
        prob.driver.add_desvar('parm.x', low=-10, high=99)
        prob.driver.add_objective('comp.f')
        prob.driver.add_constraint('comp.g', upper=0.)
        prob.setup(check=False)
        prob.run()

        indep_list = ['parm.x']
        unknown_list = ['comp.f', 'comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]), decimal=5)
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

    def test_calc_gradient_with_poi_indices(self):
        root = Group()
        root.add('parm', IndepVarComp('x', np.array([1., 1., 1., 1., 2.])))
        root.add('comp', RosenSuzuki())

        root.connect('parm.x', 'comp.x', src_indices=[0,1,2,3])

        prob = Problem(root)
        prob.driver.add_desvar('parm.x', indices=[0,1,2,3], low=-10, high=99)   # FIXME: try [0, 1, 2, 4]
        prob.driver.add_objective('comp.f')
        prob.driver.add_constraint('comp.g', upper=0.)
        prob.setup(check=False)
        prob.run()

        indep_list = ['parm.x']
        unknown_list = ['comp.f', 'comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]), decimal=5)
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

    def test_calc_gradient_with_qoi_indices(self):
        root = Group()
        root.add('parm', IndepVarComp('x', np.array([1., 1., 1., 1.])))
        root.add('comp', RosenSuzuki())

        root.connect('parm.x', 'comp.x')

        prob = Problem(root)
        prob.driver.add_desvar('parm.x', low=-10, high=99)
        prob.driver.add_objective('comp.f')
        prob.driver.add_constraint('comp.g', upper=0., indices=[0, 2])
        prob.setup(check=False)
        prob.run()

        indep_list = ['parm.x']
        unknown_list = ['comp.f', 'comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        # FIXME: currently returns an extra row of zeros
        # J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        # np.testing.assert_almost_equal(J, np.array([
        #     [-3.,  -3., -17.,  9.],
        #     [ 3.,   1.,   3.,  1.],
        #     [ 6.,   1.,   2., -1.],
        # ]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        # FIXME: currently returns an extra row of zeros
        # J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        # np.testing.assert_almost_equal(J, np.array([
        #     [-3.,  -3., -17.,  9.],
        #     [ 3.,   1.,   3.,  1.],
        #     [ 6.,   1.,   2., -1.],
        # ]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]), decimal=5)
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

    def test_calc_gradient_multiple_params(self):
        prob = Problem()
        prob.root = FanIn()
        prob.setup(check=False)
        prob.run()

        indep_list   = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))


if __name__ == "__main__":
    unittest.main()

""" Unit test for the Scipy GMRES linear solver. """

import unittest
from unittest import SkipTest

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.simplecomps import SimpleCompDerivMatVec, SimpleCompDerivJac


class TestScipyGMRES(unittest.TestCase):

    def _test_simple_matvec(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES()
        top.setup()
        top.run()

        J = top.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        self.assertAlmostEqual(J['y']['x'][0][0], 2.0, places=4)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        self.assertAlmostEqual(J['y']['x'][0][0], 2.0, places=4)

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
        self.assertAlmostEqual(J['y']['x'][0][0], 2.0, places=4)

        J = top.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        self.assertAlmostEqual(J['y']['x'][0][0], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()

""" Unit test for the Scipy GMRES linear solver. """

import unittest

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.testcomps import SimpleComp


class TestScipyGMRES(unittest.TestCase):

    def test_simple(self):

        group = Group()
        group.add('mycomp', SimpleComp(), promotes=['x', 'y'])
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])

        top = Problem()
        top.root = group
        top.root.lin_solver = ScipyGMRES(group)
        top.run()

        J = top.calc_gradient(inputs=['x'], outputs=['y'], mode='fwd')
        self.assertAlmostEqual(J[0][0], 2.0, places=4)

        J = top.calc_gradient(inputs=['x'], outputs=['y'], mode='rev')
        self.assertAlmostEqual(J[0][0], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()

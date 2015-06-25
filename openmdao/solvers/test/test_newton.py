""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.components.execcomp import ExecComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.newton import Newton
from openmdao.test.sellar import SellarDerivativesGrouped, \
                                 SellarNoDerivatives, SellarDerivatives, \
                                 SellarStateConnection
from openmdao.test.testutil import assert_rel_error


class TestNewton(unittest.TestCase):

    def test_sellar_grouped(self):

        top = Problem()
        top.root = SellarDerivativesGrouped()
        top.root.mda.nl_solver = Newton()

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)

    def test_sellar(self):

        top = Problem()
        top.root = SellarNoDerivatives()
        top.root.nl_solver = Newton()

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)

    def test_sellar_derivs(self):

        top = Problem()
        top.root = SellarDerivatives()
        top.root.nl_solver = Newton()

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)

    def test_sellar_state_connection(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.root.nl_solver = Newton()

        top.setup()
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(top.root.nl_solver.iter_count, 8)



if __name__ == "__main__":
    unittest.main()

""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.components.execcomp import ExecComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.newton import Newton
from openmdao.test.sellar import SellarDis1, SellarDis2, SellarNoDerivatives, \
                                 SellarDerivatives, SellarStateConnection
from openmdao.test.testutil import assert_rel_error

class SellarNoDerivativesGrouped(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    without derivatives."""

    def __init__(self):
        super(SellarNoDerivativesGrouped, self).__init__()

        self.add('px', ParamComp('x', 1.0), promotes=['*'])
        self.add('pz', ParamComp('z', np.array([5.0, 2.0])), promotes=['*'])
        sub = self.add('mda', Group(), promotes=['*'])

        sub.add('d1', SellarDis1(), promotes=['*'])
        sub.add('d2', SellarDis2(), promotes=['*'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, d1=0.0, d2=0.0),
                 promotes=['*'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['*'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['*'])

        sub.nl_solver = Newton()
        sub.d1.fd_options['force_fd'] = True
        sub.d2.fd_options['force_fd'] = True

class TestNewton(unittest.TestCase):

    def test_sellar_grouped(self):

        top = Problem()
        top.root = SellarNoDerivativesGrouped()

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

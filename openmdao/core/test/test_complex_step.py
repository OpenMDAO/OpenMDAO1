""" Testing out complex step capability."""

from __future__ import print_function

import unittest

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.util import assert_rel_error


class ComplexStepVectorUnitTests(unittest.TestCase):

    def test_single_comp_paraboloid(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 0.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        root.fd_options['force_fd'] = True
        root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        jac = prob.calc_gradient(['x', 'y'], ['f_xy'])

        # Note, FD can not reach this accuracy, but CS can.
        assert_rel_error(self, jac[0][0], -6.0, 1e-7)
        assert_rel_error(self, jac[0][1], 8.0, 1e-7)

    def test_converge_diverge_groups(self):

        prob = Problem()
        root = prob.root = Group()
        root.add('sub', ConvergeDivergeGroups())

        root.fd_options['force_fd'] = True
        root.fd_options['form'] = 'complex_step'

        # We can't reach our desired accuracy with this step size in fd, but
        # we can with cs.
        root.fd_options['step_size'] = 1.0e-4

        prob.setup(check=False)
        prob.run()

        indep_list = ['sub.p.x']
        unknown_list = ['sub.comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

if __name__ == "__main__":
    unittest.main()
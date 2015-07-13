""" Testing group-level finite difference. """

import unittest
import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.simplecomps import SimpleCompDerivMatVec
from openmdao.test.testutil import assert_rel_error


class TestGroupDerivatves(unittest.TestCase):

    def test_simple_matvec(self):

        class VerificationComp(SimpleCompDerivMatVec):

            def jacobian(self, params, unknowns, resids):
                raise RuntimeError("Derivative functions on this comp should not run.")

            def apply_linear(self, params, unknowns, dparams, dunknowns,
                             dresids, mode):
                raise RuntimeError("Derivative functions on this comp should not run.")

        sub = Group()
        sub.add('mycomp', VerificationComp())

        top = Problem()
        top.root = Group()
        top.root.add('sub', sub)
        top.root.add('x_param', ParamComp('x', 1.0))
        top.root.connect('x_param.x', "sub.mycomp.x")

        sub.fd_options['force_fd'] = True
        top.setup()
        top.run()

        J = top.calc_gradient(['x_param.x'], ['sub.mycomp.y'], mode='fwd',
                              return_format='dict')
        assert_rel_error(self, J['sub.mycomp.y']['x_param.x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['x_param.x'], ['sub.mycomp.y'], mode='rev',
                              return_format='dict')
        assert_rel_error(self, J['sub.mycomp.y']['x_param.x'][0][0], 2.0, 1e-6)

    def test_converge_diverge_groups(self):

        top = Problem()
        top.root = Group()
        top.root.add('sub', ConvergeDivergeGroups())

        param_list = ['sub.p.x']
        unknown_list = ['sub.comp7.y1']

        top.setup()
        top.run()

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)


if __name__ == "__main__":
    unittest.main()

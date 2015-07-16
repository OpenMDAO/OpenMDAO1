""" Testing group-level finite difference. """

import unittest
import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
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
        top.setup(check=False)
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

        top.setup(check=False)
        top.run()

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

    def test_group_fd(self):

        class SimpleComp(Component):
            """ A simple component that provides derivatives. """

            def __init__(self):
                super(SimpleComp, self).__init__()

                # Params
                self.add_param('x', 2.0)

                # Unknowns
                self.add_output('y', 0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much.  Just multiply by 3"""
                unknowns['y'] = 3.0*params['x']

            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}
                J[('y', 'x')] = 3.0
                return J


        class Model(Group):
            """ Simple model to experiment with finite difference."""

            def __init__(self):
                super(Model, self).__init__()

                self.add('px', ParamComp('x', 2.0))

                self.add('comp1', SimpleComp())
                sub = self.add('sub', Group())
                sub.add('comp2', SimpleComp())
                sub.add('comp3', SimpleComp())
                self.add('comp4', SimpleComp())

                self.connect('px.x', 'comp1.x')
                self.connect('comp1.y', 'sub.comp2.x')
                self.connect('sub.comp2.y', 'sub.comp3.x')
                self.connect('sub.comp3.y', 'comp4.x')

                self.sub.fd_options['force_fd'] = True

        top = Problem()
        top.root = Model()

        top.setup(check=False)
        top.run()

        J = top.calc_gradient(['px.x'], ['comp4.y'])
        assert_rel_error(self, J[0][0], 81.0, 1e-6)

if __name__ == "__main__":
    unittest.main()

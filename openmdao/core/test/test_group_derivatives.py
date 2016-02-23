""" Testing group-level finite difference. """

import unittest
import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.simple_comps import SimpleCompDerivMatVec
from openmdao.test.util import assert_rel_error


class TestGroupDerivatves(unittest.TestCase):

    def test_simple_matvec(self):

        class VerificationComp(SimpleCompDerivMatVec):

            def linearize(self, params, unknowns, resids):
                raise RuntimeError("Derivative functions on this comp should not run.")

            def apply_linear(self, params, unknowns, dparams, dunknowns,
                             dresids, mode):
                raise RuntimeError("Derivative functions on this comp should not run.")

        sub = Group()
        sub.add('mycomp', VerificationComp())

        prob = Problem()
        prob.root = Group()
        prob.root.add('sub', sub)
        prob.root.add('x_param', IndepVarComp('x', 1.0))
        prob.root.connect('x_param.x', "sub.mycomp.x")

        sub.fd_options['force_fd'] = True
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x_param.x'], ['sub.mycomp.y'], mode='fwd',
                              return_format='dict')
        assert_rel_error(self, J['sub.mycomp.y']['x_param.x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x_param.x'], ['sub.mycomp.y'], mode='rev',
                              return_format='dict')
        assert_rel_error(self, J['sub.mycomp.y']['x_param.x'][0][0], 2.0, 1e-6)

    def test_converge_diverge_groups(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('sub', ConvergeDivergeGroups())

        indep_list = ['sub.p.x']
        unknown_list = ['sub.comp7.y1']

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
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

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}
                J[('y', 'x')] = 3.0
                return J

        class Model(Group):
            """ Simple model to experiment with finite difference."""

            def __init__(self):
                super(Model, self).__init__()

                self.add('px', IndepVarComp('x', 2.0))

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

        prob = Problem()
        prob.root = Model()

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['px.x'], ['comp4.y'])
        assert_rel_error(self, J[0][0], 81.0, 1e-6)

    def test_group_fd_pbo(self):

        # Bug report and test from Ry10.

        class C1(Component):
            def __init__(self):
                super(C1, self).__init__()
                self.add_param('x', shape=1)
                self.add_param('B', val=0, pass_by_obj=True)
                self.add_output('y', shape=1)
                self.add_output('BB', val=0, pass_by_obj=True)
                self.fd_options['force_fd'] = True

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['y'] = 4.0*params['x']*params['B']

        class C2(Component):
            def __init__(self):
                super(C2, self).__init__()
                self.add_param('y', shape=1)
                self.add_output('z', shape=1)
                self.fd_options['force_fd'] = True

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['z'] = 2.0*params['y']

        class FDGroup(Group):
            def __init__(self):
                super(FDGroup, self).__init__()
                self.add('c1', C1(), promotes=['*'])
                self.add('c2', C2(), promotes=['*'])
                self.fd_options['force_fd'] = True # Comment and then it works

        class RootGroup(Group):
            def __init__(self):
                super(RootGroup, self).__init__()
                self.add('x', IndepVarComp('x', 0.0), promotes=['*'])
                self.add('B', IndepVarComp('B', 0, pass_by_obj=True), promotes=['*'])
                self.add('fd_group', FDGroup(), promotes=['*'])

        p = Problem()
        p.root = RootGroup()
        p.setup(check=False)
        p['x'] = 1.5
        p['B'] = 2
        p.run()

        J = p.calc_gradient(['x'], ['y', 'z'], mode='fwd')
        assert_rel_error(self, J[0][0], 8.0, 1e-6)
        assert_rel_error(self, J[1][0], 16.0, 1e-6)

if __name__ == "__main__":
    unittest.main()

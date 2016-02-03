""" Testing out complex step capability."""

from __future__ import print_function

import unittest

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.core.test.test_units import SrcComp, TgtCompC, TgtCompF, TgtCompK
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.util import assert_rel_error

try:
    from openmdao.solvers.petsc_ksp import PetscKSP
    from openmdao.core.petsc_impl import PetscImpl as petsc_impl
except ImportError:
    petsc_impl = None


class ComplexStepVectorUnitTestsBasicImpl(unittest.TestCase):

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
        # we can with cs (where step size is irrelevant.)
        root.fd_options['step_size'] = 1.0e-4

        prob.setup(check=False)
        prob.run()

        indep_list = ['sub.p.x']
        unknown_list = ['sub.comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

    def test_unit_conversion(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)


class ComplexStepVectorUnitTestsPETSCImpl(unittest.TestCase):

    def test_single_comp_paraboloid(self):
        prob = Problem(impl=petsc_impl)
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

        prob = Problem(impl=petsc_impl)
        root = prob.root = Group()
        root.add('sub', ConvergeDivergeGroups())

        root.fd_options['force_fd'] = True
        root.fd_options['form'] = 'complex_step'

        # We can't reach our desired accuracy with this step size in fd, but
        # we can with cs (where step size is irrelevant.)
        root.fd_options['step_size'] = 1.0e-4

        prob.setup(check=False)
        prob.run()

        indep_list = ['sub.p.x']
        unknown_list = ['sub.comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp7.y1']['sub.p.x'][0][0], -40.75, 1e-6)

    def test_unit_conversion(self):

        prob = Problem(impl=petsc_impl)
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)


if __name__ == "__main__":
    unittest.main()

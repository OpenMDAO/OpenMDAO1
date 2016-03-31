""" Testing out complex step capability."""

from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Group, Problem, IndepVarComp, Newton, UnitComp, ExecComp, \
                         Component
from openmdao.core.test.test_units import SrcComp, TgtCompC, TgtCompF, TgtCompK
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivativesGrouped
from openmdao.test.util import assert_rel_error
from openmdao.util.options import OptionsDictionary

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

    def test_sellar_derivs_grouped(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.root.mda.nl_solver.options['atol'] = 1e-12
        prob.setup(check=False)
        prob.run()

        # Just make sure we are at the right answer
        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        indep_list = ['x', 'z']
        unknown_list = ['obj', 'con1', 'con2']

        Jbase = {}
        Jbase['con1'] = {}
        Jbase['con1']['x'] = -0.98061433
        Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
        Jbase['con2'] = {}
        Jbase['con2']['x'] = 0.09692762
        Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
        Jbase['obj'] = {}
        Jbase['obj']['x'] = 2.98061392
        Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)


        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        prob.root.fd_options['form'] = 'central'

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

    def test_complex_step_around_newton_error(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.root.mda.nl_solver = Newton()

        with self.assertRaises(RuntimeError) as cm:
            prob.setup(check=False)

        msg = "The solver in 'mda' requires derivatives. We "
        msg += "currently do not support complex step around it."

        self.assertTrue(msg in str(cm.exception))

    def test_sub_unsupported(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        # We don't support submodel cs yet.
        prob.root.mda.fd_options['force_fd'] = True
        prob.root.mda.fd_options['form'] = 'complex_step'

        with self.assertRaises(RuntimeError) as cm:
            prob.setup(check=False)

        msg = "Complex step is currently not supported for groups"
        msg += " other than root."

        self.assertTrue(msg in str(cm.exception))

    def test_array_values_diff_shape_units(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', np.zeros((2, 3)), units='degC'), promotes=['x'])
        prob.root.add('uc', UnitComp(shape=(2, 3), param_name='x', out_name='x_out', units='degF'),
                      promotes=['x', 'x_out'])

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(6), 1e-6)

    def test_array_values_diff_shape_no_units(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', np.zeros((2, 3))), promotes=['x'])
        prob.root.add('uc', ExecComp('x_out = 1.8*x', x=np.zeros((2, 3)), x_out=np.zeros((2, 3))),
                      promotes=['x', 'x_out'])

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'], 1.8*np.eye(6), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'], 1.8*np.eye(6), 1e-6)

    def test_array_values_same_shape_units(self):
        prob = Problem()
        prob.root = Group()
        prob.root.add('pc', IndepVarComp('x', np.zeros((2, )), units='degC'), promotes=['x'])
        prob.root.add('uc', UnitComp(shape=(2, ), param_name='x', out_name='x_out', units='degF'),
                      promotes=['x', 'x_out'])

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['x_out']

        # Forward Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(2), 1e-6)

        # Reverse Mode
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')
        assert_rel_error(self, J['x_out']['x'],1.8*np.eye(2), 1e-6)

    def test_single_comp_paraboloid_pbo_hanging_param(self):

        class ParaboloidPBO(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(ParaboloidPBO, self).__init__()

                self.add_param('x', val=0.0)
                self.add_param('y', val=0.0, pass_by_obj=True)

                self.add_output('f_xy', val=0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Optimal solution (minimum): x = 6.6667; y = -7.3333
                """

                x = params['x']
                y = params['y']

                unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 0.0, pass_by_obj=True), promotes=['*'])
        root.add('comp', ParaboloidPBO(), promotes=['x', 'y', 'f_xy'])

        root.fd_options['force_fd'] = True
        root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        jac = prob.calc_gradient(['x'], ['f_xy'])

        # Note, FD can not reach this accuracy, but CS can.
        assert_rel_error(self, jac[0][0], -6.0, 1e-7)


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

    def test_sellar_derivs_grouped(self):

        prob = Problem(impl=petsc_impl)
        prob.root = SellarDerivativesGrouped()

        prob.root.fd_options['force_fd'] = True
        prob.root.fd_options['form'] = 'complex_step'

        prob.root.mda.nl_solver.options['atol'] = 1e-12
        prob.setup(check=False)
        prob.run()

        # Just make sure we are at the right answer
        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        indep_list = ['x', 'z']
        unknown_list = ['obj', 'con1', 'con2']

        Jbase = {}
        Jbase['con1'] = {}
        Jbase['con1']['x'] = -0.98061433
        Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
        Jbase['con2'] = {}
        Jbase['con2']['x'] = 0.09692762
        Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
        Jbase['obj'] = {}
        Jbase['obj']['x'] = 2.98061392
        Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

if __name__ == "__main__":
    unittest.main()

""" Unit test for the Scipy GMRES linear solver. """

import unittest
import numpy as np

from openmdao.api import Group, Problem, IndepVarComp, ScipyGMRES, \
    DirectSolver, ExecComp, LinearGaussSeidel, AnalysisError
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.sellar import SellarDerivativesGrouped
from openmdao.test.simple_comps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                       FanOutGrouped, DoubleArrayComp, \
                                       FanInGrouped, ArrayComp2D, FanOutAllGrouped
from openmdao.test.util import assert_rel_error
from openmdao.util.options import OptionsDictionary


class TestScipyGMRES(unittest.TestCase):

    def test_simple_matvec(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = Group()
        prob.root.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        prob.root.add('sub', group, promotes=['*'])

        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed_like_multipoint(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = Group()
        prob.root.add('sub', group, promotes=['*'])
        prob.root.sub.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])

        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='array')
        assert_rel_error(self, J[0][0], 2.0, 1e-6)

    def test_array2D(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = prob.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_double_arraycomp(self):
        # Mainly testing a bug in the array return for multiple arrays

        group = Group()
        group.add('x_param1', IndepVarComp('x1', np.ones((2))), promotes=['*'])
        group.add('x_param2', IndepVarComp('x2', np.ones((2))), promotes=['*'])
        group.add('mycomp', DoubleArrayComp(), promotes=['*'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        Jbase = group.mycomp.JJ

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fwd',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fd',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='rev',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_jac(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', ExecComp(['y=2.0*x']), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_fan_out(self):

        prob = Problem()
        prob.root = FanOut()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['comp2.y', "comp3.y"]

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_out_grouped(self):

        prob = Problem()
        prob.root = FanOutGrouped()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['sub.comp2.y', "sub.comp3.y"]

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_in(self):

        prob = Problem()
        prob.root = FanIn()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_grouped(self):

        prob = Problem()
        prob.root = FanInGrouped()
        prob.root.ln_solver = ScipyGMRES()

        indep_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_converge_diverge(self):

        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['comp7.y1']

        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    def test_analysis_error(self):

        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.ln_solver.options['maxiter'] = 2
        prob.root.ln_solver.options['err_on_maxiter'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['comp7.y1']

        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

        try:
            J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': ScipyGMRES failed to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_converge_diverge_groups(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

        indep_list = ['p.x']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    def test_single_diamond(self):

        prob = Problem()
        prob.root = SingleDiamond()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['comp4.y1', 'comp4.y2']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    def test_single_diamond_grouped(self):

        prob = Problem()
        prob.root = SingleDiamondGrouped()
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['comp4.y1', 'comp4.y2']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    def test_sellar_derivs_grouped(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

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

    def test_generate_numpydocstring(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()

        test_string = prob.root.ln_solver.generate_docstring()

        original_string = \
"""    \"\"\"

    Options
    -------
    options['atol'] : float(1e-12)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] : int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] : int(1000)
        Maximum number of iterations.
    options['mode'] : str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['restart'] : int(20)
        Number of iterations between restarts. Larger values increase iteration cost, but may be necessary for convergence

    \"\"\"
"""
        for sorig, stest in zip(original_string.split('\n'), test_string.split('\n')):
            self.assertEqual(sorig, stest)


class TestScipyGMRESPreconditioner(unittest.TestCase):

    def test_sellar_derivs_grouped_precon(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.root.mda.nl_solver.options['atol'] = 1e-12
        prob.root.ln_solver.preconditioner = LinearGaussSeidel()
        prob.root.mda.ln_solver = DirectSolver()
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

    def test_converge_diverge_groups(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.ln_solver.preconditioner = LinearGaussSeidel()

        prob.root.sub1.ln_solver = DirectSolver()
        prob.root.sub3.ln_solver = DirectSolver()

        prob.setup(check=False)
        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

        indep_list = ['p.x']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    def test_fan_out_all_grouped(self):

        prob = Problem()
        prob.root = FanOutAllGrouped()
        prob.root.ln_solver = ScipyGMRES()

        prob.root.ln_solver.preconditioner = LinearGaussSeidel()
        prob.root.sub1.ln_solver = DirectSolver()
        prob.root.sub2.ln_solver = DirectSolver()
        prob.root.sub3.ln_solver = DirectSolver()

        prob.setup(check=False)
        prob.run()

        indep_list = ['p.x']
        unknown_list = ['sub2.comp2.y', "sub3.comp3.y"]

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['sub2.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub3.comp3.y']['p.x'][0][0], 15.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['sub2.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub3.comp3.y']['p.x'][0][0], 15.0, 1e-6)


if __name__ == "__main__":
    unittest.main()

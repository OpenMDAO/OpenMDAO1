""" Unit test for the Gauss Seidel linear solver. """

import unittest
import numpy as np

from openmdao.api import Group, Problem, LinearGaussSeidel, IndepVarComp, ExecComp
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.sellar import SellarDerivativesGrouped, SellarDerivatives, StateConnection
from openmdao.test.simple_comps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                       FanOutGrouped, FanInGrouped, ArrayComp2D
from openmdao.test.util import assert_rel_error


class TestLinearGaussSeidel(unittest.TestCase):

    def test_simple_matvec(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
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

        prob.root.ln_solver = LinearGaussSeidel()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_array2D(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = prob.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_array2D_index_connection(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        sub = group.add('sub', Group(), promotes=['*'])
        sub.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])
        group.add('obj', ExecComp('b = a'))
        group.connect('y', 'obj.a',  src_indices=[3])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['obj.b'], mode='fwd', return_format='dict')
        Jbase = prob.root.sub.mycomp._jacobian_cache
        assert_rel_error(self, Jbase[('y', 'x')][3][0], J['obj.b']['x'][0][0], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][1], J['obj.b']['x'][0][1], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][2], J['obj.b']['x'][0][2], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][3], J['obj.b']['x'][0][3], 1e-8)

        J = prob.calc_gradient(['x'], ['obj.b'], mode='rev', return_format='dict')
        Jbase = prob.root.sub.mycomp._jacobian_cache
        assert_rel_error(self, Jbase[('y', 'x')][3][0], J['obj.b']['x'][0][0], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][1], J['obj.b']['x'][0][1], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][2], J['obj.b']['x'][0][2], 1e-8)
        assert_rel_error(self, Jbase[('y', 'x')][3][3], J['obj.b']['x'][0][3], 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_two_simple(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0))
        group.add('comp1', ExecComp(['y=2.0*x']))
        group.add('comp2', ExecComp(['z=3.0*y']))

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.connect('x_param.x', 'comp1.x')
        prob.root.connect('comp1.y', 'comp2.y')

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x_param.x'], ['comp2.z'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp2.z']['x_param.x'][0][0], 6.0, 1e-6)

        J = prob.calc_gradient(['x_param.x'], ['comp2.z'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp2.z']['x_param.x'][0][0], 6.0, 1e-6)

    def test_fan_out(self):

        prob = Problem()
        prob.root = FanOut()
        prob.root.ln_solver = LinearGaussSeidel()
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
        prob.root.ln_solver = LinearGaussSeidel()
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

    def test_fan_out_grouped_GS_GS(self):

        prob = Problem()
        prob.root = FanOutGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()
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
        prob.root.ln_solver = LinearGaussSeidel()
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
        prob.root.ln_solver = LinearGaussSeidel()
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

    def test_fan_in_grouped_GS_GS(self):

        prob = Problem()
        prob.root = FanInGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.sub.ln_solver = LinearGaussSeidel()
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

    def test_converge_diverge(self):

        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.ln_solver = LinearGaussSeidel()
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

    def test_converge_diverge_groups(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()
        prob.root.ln_solver = LinearGaussSeidel()
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
        prob.root.ln_solver = LinearGaussSeidel()
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
        prob.root.ln_solver = LinearGaussSeidel()
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

    def test_sellar_derivs(self):

        prob = Problem()
        prob.root = SellarDerivatives()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['maxiter'] = 10
        prob.root.ln_solver.options['atol'] = 1e-12
        prob.root.ln_solver.options['rtol'] = 1e-12
        #prob.root.ln_solver.options['iprint'] = 1

        prob.root.nl_solver.options['atol'] = 1e-12
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

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

        # Obviously this test doesn't do much right now, but I need to verify
        # we don't get a keyerror here.
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')

    def test_sellar_derivs_grouped(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['maxiter'] = 15
        #prob.root.ln_solver.options['iprint'] = 1

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

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

    def test_sellar_derivs_grouped_GSnested(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()
        prob.root.ln_solver = LinearGaussSeidel()
        prob.root.ln_solver.options['maxiter'] = 15
        #prob.root.ln_solver.options['iprint'] = 1

        prob.root.mda.nl_solver.options['atol'] = 1e-12
        prob.root.mda.ln_solver = LinearGaussSeidel()
        prob.root.mda.ln_solver.options['maxiter'] = 15
        #prob.root.mda.ln_solver.options['iprint'] = 1
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

        prob.root.fd_options['form'] = 'central'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J[key1][key2], val2, .00001)

    def test_lings_cycle_msg(self):
        p = Problem(root=Group())
        root = p.root
        C1 = root.add("C1", ExecComp('y=x*2.0'))
        C2 = root.add("C2", ExecComp('y=x*2.0'))
        C3 = root.add("C3", ExecComp('y=x*2.0'))

        root.connect('C1.y', 'C3.x')
        root.connect('C3.y', 'C2.x')
        root.connect('C2.y', 'C1.x')

        try:
            p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "Group '' has a LinearGaussSeidel solver with maxiter==1 but it contains cycles "
                             "[['C1', 'C2', 'C3']]. To fix this error, change to a different linear solver, "
                             "e.g. ScipyGMRES or PetscKSP, or increase maxiter (not recommended).")
        else:
            self.fail("Exception expected")

    def test_lings_state_msg(self):
        p = Problem(root=Group())
        root = p.root
        C1 = root.add("C1", StateConnection())

        try:
            p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "Group '' has a LinearGaussSeidel solver with maxiter==1 but it contains "
                             "implicit states ['C1.y2_command']. To fix this error, change to a different "
                             "linear solver, e.g. ScipyGMRES or PetscKSP, or increase maxiter (not recommended).")
        else:
            self.fail("Exception expected")


if __name__ == "__main__":
    unittest.main()

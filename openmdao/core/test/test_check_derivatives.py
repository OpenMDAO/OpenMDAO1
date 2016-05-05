""" Testing for Problem.check_partial_derivatives and check_total_derivatives."""

import unittest
from six import iteritems, StringIO, PY3
from six.moves import cStringIO as StringIO

import numpy as np

from openmdao.api import Group, Component, IndepVarComp, Problem, ScipyGMRES, \
                          ParallelGroup, LinearGaussSeidel
from openmdao.test.converge_diverge import ConvergeDivergeGroups, ConvergeDivergePar
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simple_comps import SimpleArrayComp, SimpleImplicitComp, \
                                      SimpleCompDerivMatVec
from openmdao.test.util import assert_rel_error
from openmdao.util.options import OptionsDictionary


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        self.assertEqual(len(data), 7)

        # Piggyback a test for the 'comps' option.

        data = prob.check_partial_derivatives(out_stream=None,
                                              comps=['sub1.sub2.comp3', 'comp7'])
        self.assertEqual(len(data), 2)
        self.assertTrue('sub1.sub2.comp3' in data)
        self.assertTrue('comp7' in data)

        with self.assertRaises(RuntimeError) as cm:
            data = prob.check_partial_derivatives(out_stream=None,
                                                  comps=['sub1', 'bogus'])

        expected_msg = "The following are not valid comp names: ['bogus', 'sub1']"
        self.assertEqual(str(cm.exception), expected_msg)

        # This is a good test to piggyback the compact_print test

        mystream = StringIO()
        prob.check_partial_derivatives(out_stream=mystream,
                                       compact_print=True)

        text = mystream.getvalue()
        expected = "'y1'            wrt 'x1'            | 8.0000e+00 | 8.0000e+00 |  8.0000e+00 | 2.0013e-06 | 2.0013e-06 | 2.5016e-07 | 2.5016e-07"
        self.assertTrue(expected in text)


    def test_double_diamond_model_complex_step(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.root.sub1.comp1.fd_options['form'] = 'complex_step'
        prob.root.sub1.sub2.comp2.fd_options['form'] = 'complex_step'
        prob.root.sub1.sub2.comp3.fd_options['form'] = 'complex_step'
        prob.root.sub1.comp4.fd_options['form'] = 'complex_step'
        prob.root.sub3.comp5.fd_options['form'] = 'complex_step'
        prob.root.sub3.comp6.fd_options['form'] = 'complex_step'
        prob.root.comp7.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleArrayComp())
        prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit_run_once(self):

        class SIC2(SimpleImplicitComp):

            def solve_nonlinear(self, params, unknowns, resids):
                """ Simple iterative solve. (Babylonian method)."""

                super(SIC2, self).solve_nonlinear(params, unknowns, resids)

                # This mimics a problem with residuals that aren't up-to-date
                # with the solve
                resids['z'] = 999.999


        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SIC2())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run_once()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit_complex_step(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.comp.fd_options['step_size'] = 1.0e4
        prob.root.comp.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_bad_size(self):

        class BadComp(SimpleArrayComp):
            def linearize(self, params, unknowns, resids):
                """Analytical derivatives"""
                J = {}
                J[('y', 'x')] = np.zeros((3, 3))
                return J


        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BadComp())
        prob.root.add('p1', IndepVarComp('x', np.ones([2])))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        try:
            data = prob.check_partial_derivatives(out_stream=None)
        except Exception as err:
            msg = "derivative in component 'comp' of 'y' wrt 'x' is the wrong size. It should be (2, 2), but got (3, 3)"

            # Good ole Numpy
            raised_error = str(err)
            raised_error = raised_error.replace('3L', '3')

            self.assertEqual(raised_error, msg)
        else:
            self.fail("Error expected")

    def test_incorrect_jacobian(self):

        class MyComp(Component):

            def __init__(self, multiplier=2.0):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 5.0)

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 3.0*params['x1'] + 4.0*params['x2']

            def linearize(self, params, unknowns, resids):
                """Intentionally incorrect derivative."""

                J = {}
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])
                return J

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp('x1', 3.0))
        prob.root.add('p2', IndepVarComp('x2', 5.0))

        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run()

        stringstream = StringIO()

        data = prob.check_partial_derivatives(out_stream=stringstream)

        lines = stringstream.getvalue().split("\n")

        y_wrt_x1_line = lines.index("  comp: 'y' wrt 'x1'")

        self.assertTrue(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+7].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertFalse(lines[y_wrt_x1_line+8].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_big_boy_Jacobian(self):

        class MyComp(Component):

            def __init__(self, multiplier=2.0):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 5.0)

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 3.0*params['x1'] + 4.0*params['x2']

            def linearize(self, params, unknowns, resids):
                """Intentionally left out x2 derivative."""

                J = {}
                J[('y', 'x1')] = np.array([[3.0]])
                return J

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp('x1', 3.0))
        prob.root.add('p2', IndepVarComp('x2', 5.0))

        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)
        self.assertTrue(('y', 'x1') in data['comp'])
        self.assertTrue(('y', 'x2') in data['comp'])

    def test_extra_fd(self):

        class CSTestComp(Component):

            def __init__(self):
                super(CSTestComp, self).__init__()

                self.add_param('x', val=1.5, step_size=1e-2) # pick a big step to make sure FD sucks
                self.add_output('f', val=0.)

            def solve_nonlinear(self, p, u, r):
                x = p['x']
                u['f'] = np.exp(x)/np.sqrt(np.sin(x)**3 + np.cos(x)**3)

        p = Problem()
        p.root = Group()

        p.root.add('des_vars', IndepVarComp('x', 1.5), promotes=['*'])
        c = p.root.add('comp', CSTestComp(), promotes=["*"])
        c.fd_options['force_fd'] = True
        c.fd_options['form'] = "complex_step"
        c.fd_options['extra_check_partials_form'] = "forward"

        p.setup(check=False)
        p.run_once()

        check_data = p.check_partial_derivatives(out_stream=None)
        cs_val = check_data['comp']['f','x']['J_fd'][0,0] # should be the complex steped value!
        assert_rel_error(self, cs_val, 4.05289181447, 1e-8)

        fd2_val = check_data['comp']['f','x']['J_fd2'][0,0] # should be the real-fd'd value!
        assert_rel_error(self, fd2_val, 4.10128351131, 1e-8)

        # For coverage

        mystream = StringIO()
        p.check_partial_derivatives(out_stream=mystream, compact_print=True)

        text = mystream.getvalue()
        expected = "'f'             wrt 'x'             |  4.052892e+00 | 4.101284e+00 |  4.839170e-02 |  1.194004e-02"
        self.assertTrue(expected in text)

class TestProblemFullFD(unittest.TestCase):

    def test_full_model_fd_simple_comp(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', SimpleCompDerivMatVec())
        prob.root.add('p1', IndepVarComp('x', 1.0))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['comp.x']
        unknown_list = ['comp.y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['comp.x'][0][0], 2.0, 1e-6)

        # We should not allocate deriv vectors for full model FD
        self.assertEqual(len(prob.root.dumat[None].vec), 0)
        self.assertEqual(len(prob.root.drmat[None].vec), 0)
        self.assertEqual(len(prob.root.dpmat[None].vec), 0)

    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['x']
        unknown_list = ['y']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        # We should not allocate deriv vectors for full model FD
        self.assertEqual(len(prob.root.dumat[None].vec), 0)
        self.assertEqual(len(prob.root.drmat[None].vec), 0)
        self.assertEqual(len(prob.root.dpmat[None].vec), 0)

    def test_full_model_fd_double_diamond_grouped(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        indep_list = ['sub1.comp1.x1']
        unknown_list = ['comp7.y1']

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)

        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        prob.root.fd_options['form'] = 'central'

        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7.y1']['sub1.comp1.x1'][0][0], -40.75, 1e-6)

    def test_full_model_fd_double_diamond_grouped_par_sys(self):

        prob = Problem()
        root = prob.root = Group()
        par = root.add('par', ParallelGroup())
        par.add('sub', ConvergeDivergeGroups())

        prob.root.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        # Make sure we don't get a key error.
        data = prob.check_total_derivatives(out_stream=None)


class TestProblemCheckTotals(unittest.TestCase):

    def test_double_diamond_model(self):

        prob = Problem()
        prob.root = ConvergeDivergeGroups()

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_full_model_fd_simple_comp_promoted(self):

        prob = Problem()
        prob.root = Group()
        sub = prob.root.add('sub', Group(), promotes=['*'])
        sub.add('comp', SimpleCompDerivMatVec(), promotes=['*'])
        prob.root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][2], 0.0, 1e-5)

    def test_limit_to_desvar_obj_con(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver.add_desvar('x')
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)
        self.assertTrue(('f_xy', 'x') in data)
        self.assertTrue(('f_xy', 'y') not in data)

    def test_with_relevance_fwd(self):

        prob = Problem()
        prob.root = ConvergeDivergePar()
        prob.root.ln_solver = LinearGaussSeidel()

        prob.driver.add_desvar('p.x')
        prob.driver.add_objective('comp7.y1')
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-5)

    def test_with_relevance_rev(self):

        prob = Problem()
        prob.root = ConvergeDivergePar()
        prob.root.ln_solver = LinearGaussSeidel()

        prob.driver.add_desvar('p.x')
        prob.driver.add_objective('comp7.y1')
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        prob.setup(check=False)
        prob.run()

        data = prob.check_total_derivatives(out_stream=None)

        for key, val in iteritems(data):
            assert_rel_error(self, val['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val['rel error'][1], 0.0, 1e-5)

    def test_check_partials_calls_run_once(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 1.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver.add_desvar('x')
        prob.driver.add_desvar('y')
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)

        prob['x'] = 5.0
        prob['y'] = 2.0

        iostream = StringIO()

        data = prob.check_partial_derivatives(out_stream=iostream)

        self.assertAlmostEqual(first=prob["f_xy"],
                               second= (prob['x']-3.0)**2 \
                                       + prob['x']*prob['y'] \
                                       + (prob['y']+4.0)**2 - 3.0,
                               places=5,
                               msg="check partial derivatives did not call"
                                   "run_once on the driver as expected.")

        self.assertEqual(first=iostream.getvalue()[:39],
                         second="Executing model to populate unknowns...",
                         msg="check partial derivatives failed to run driver once")

    def test_check_totals_calls_run_once(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 1.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 1.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver.add_desvar('x')
        prob.driver.add_desvar('y')
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)

        prob['x'] = 5.0
        prob['y'] = 2.0

        iostream = StringIO()

        data = prob.check_total_derivatives(out_stream=iostream)

        self.assertAlmostEqual(first=prob["f_xy"],
                               second= (prob['x']-3.0)**2 \
                                       + prob['x']*prob['y'] \
                                       + (prob['y']+4.0)**2 - 3.0,
                               places=5,
                               msg="check partial derivatives did not call"
                                   "run_once on the driver as expected.")

        self.assertEqual(first=iostream.getvalue()[:39],
                         second="Executing model to populate unknowns...",
                         msg="check partial derivatives failed to run driver once")

if __name__ == "__main__":
    unittest.main()

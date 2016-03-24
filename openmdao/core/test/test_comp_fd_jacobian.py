""" Unit test for the fd_jacobian method on Component. This method peforms a
finite difference."""

from __future__ import print_function
from collections import OrderedDict
import unittest

import numpy as np

from openmdao.api import Component, Group, Problem, System, \
    IndepVarComp, ExecComp, ScipyGMRES
from openmdao.core.vec_wrapper import SrcVecWrapper
from openmdao.test.simple_comps import SimpleArrayComp, \
                                      SimpleImplicitComp
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.util import assert_equal_jacobian, assert_rel_error
from openmdao.util.options import OptionsDictionary


class FDpropsComp(Component):

    def __init__(self, **kwargs):
        super(FDpropsComp, self).__init__()

        # Params
        self.add_param('x1', 3.0)
        self.add_param('x2', 3.0, **kwargs)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """
        unknowns['y'] = 7.0*params['x1']**2 + 7.0*params['x2']**2


class TestProb(Problem):

    def __init__(self):
        super(TestProb, self).__init__()

        self.root = root = Group()
        root.ln_solver = ScipyGMRES()
        root.add('c1', SimpleArrayComp())
        root.add('p1', IndepVarComp('p', 1*np.ones(2)))
        root.connect('p1.p','c1.x')

        root.add('ci1', SimpleImplicitComp())
        root.add('pi1', IndepVarComp('p', 1.))
        root.connect('pi1.p','ci1.x')

        root.add('pjunk', IndepVarComp('pj', np.ones((2,2))))
        root.add('junk', ExecComp('y=x', x=np.zeros((2,2)), y=np.zeros((2,2))))
        root.connect('pjunk.pj', 'junk.x')


class CompFDTestCase(unittest.TestCase):
    """ Some basic tests of the fd_jacobian method in Component."""

    def setUp(self):
        self.p = TestProb()
        self.p.setup(check=False)

    def test_correct_keys_in_jac(self):

        expected_keys=[('y', 'x')]

        params = self.p.root.c1.params
        unknowns = self.p.root.c1.unknowns
        resids = self.p.root.c1.resids

        jac = self.p.root.c1.fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

    def test_correct_vals_in_jac(self):

        #params_dict = OrderedDict()
        #params_dict['x'] = { 'val': np.ones((2)),
                             #'pathname' : 'x',
                             #'shape': 2, 'size' : 2 }

        #unknowns_dict = OrderedDict()
        #unknowns_dict['y'] = { 'val': np.zeros((2)),
                               #'pathname' : 'y',
                             #'shape': 2, 'size' : 2 }

        #resids_dict = OrderedDict()
        #resids_dict['y'] = { 'val': np.zeros((2)),
                             #'pathname' : 'y',
                             #'shape': 2, 'size' : 2 }

        #params = SrcVecWrapper()
        #params.setup(params_dict, store_byobjs=True)

        #unknowns = SrcVecWrapper()
        #unknowns.setup(unknowns_dict, store_byobjs=True)

        #resids = SrcVecWrapper()
        #resids.setup(resids_dict, store_byobjs=True)

        params = self.p.root.c1.params
        unknowns = self.p.root.c1.unknowns
        resids = self.p.root.c1.resids
        self.p.root.c1.solve_nonlinear(params, unknowns, resids)

        jac = self.p.root.c1.fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2.,  7.],[ 5., -3.]])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

        #Got lucky that the way this comp was written, it would accept any square
        # matrix. But provided jacobian would be really wrong!

        params = self.p.root.junk.params
        unknowns = self.p.root.junk.unknowns
        resids = self.p.root.junk.resids

        params['x'] = np.ones((2,2))
        unknowns['y'] = np.zeros((2,2))
        resids['y'] = np.zeros((2,2))

        self.p.root.c1.solve_nonlinear(params, unknowns, resids)

        jac = self.p.root.c1.fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2,  0,  7,  0],
                                              [ 0,  2,  0,  7],
                                              [ 5,  0, -3,  0],
                                              [ 0,  5,  0, -3]
                                             ])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

    def test_correct_vals_in_jac_implicit(self):

        # Partials

        params = self.p.root.ci1.params
        unknowns = self.p.root.ci1.unknowns
        resids = self.p.root.ci1.resids

        params['x'] = np.array([0.5])
        unknowns['y'] = np.array([0.0])
        unknowns['z'] = np.array([0.0])
        resids['y'] = np.array([0.0])
        resids['z'] = np.array([0.0])

        self.p.root.ci1.solve_nonlinear(params, unknowns, resids)

        jac = self.p.root.ci1.fd_jacobian(params, unknowns, resids)
        expected_jac = {}
        # Output equation
        expected_jac[('y', 'x')] = 1.
        expected_jac[('y', 'z')] = 2.0
        # State equation
        expected_jac[('z', 'z')] = params['x'] + 1
        expected_jac[('z', 'x')] = unknowns['z']

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

        # Totals

        # Really tighten this up
        self.p.root.ci1.atol = 1e-14
        self.p.root.ci1.solve_nonlinear(params, unknowns, resids)

        jac = self.p.root.ci1.fd_jacobian(params, unknowns, resids, total_derivs=True)
        expected_jac = {}
        expected_jac[('y', 'x')] = -2.5555555555555554
        expected_jac[('z', 'x')] = -1.7777777777777777

        assert_equal_jacobian(self, jac, expected_jac, 1e-5)

    def test_override_states(self):

        expected_keys=[('y', 'x'), ('y', 'z'), ('z', 'x'), ('z', 'z')]

        params = self.p.root.ci1.params
        unknowns = self.p.root.ci1.unknowns
        resids = self.p.root.ci1.resids

        jac = self.p.root.ci1.fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

        # Don't compute derivs wrt 'z'
        expected_keys=[('y', 'x'), ('z', 'x')]

        params = self.p.root.ci1.params
        unknowns = self.p.root.ci1.unknowns
        resids = self.p.root.ci1.resids

        jac = self.p.root.ci1.fd_jacobian(params, unknowns, resids, fd_states=[])
        self.assertEqual(set(expected_keys), set(jac.keys()))


class CompFDinSystemTestCase(unittest.TestCase):
    """ Tests automatic finite difference of a component in a full problem."""

    def test_no_derivatives(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', ExecComp('y=x*2.0'))
        prob.root.add('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.x')

        comp.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'][0][0], 2.0, 1e-6)

    def test_overrides(self):

        class OverrideComp(Component):

            def __init__(self):
                super(OverrideComp, self).__init__()

                # Params
                self.add_param('x', 3.0)

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 7.0*params['x']

            def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                             mode):
                """Never Call."""
                raise RuntimeError("This should have been overridden by force_fd.")

            def linearize(self, params, unknowns, resids):
                """Never Call."""
                raise RuntimeError("This should have been overridden by force_fd.")

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', OverrideComp())
        prob.root.add('p1', IndepVarComp('x', 2.0))
        prob.root.connect('p1.x', 'comp.x')

        comp.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp.y']['p1.x'][0][0], 7.0, 1e-6)

    def test_fd_options_step_size(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', Paraboloid())
        prob.root.add('p1', IndepVarComp([('x', 15.0), ('y', 15.0)]))
        prob.root.connect('p1.x', 'comp.x')
        prob.root.connect('p1.y', 'comp.y')

        comp.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        assert_rel_error(self, J['comp.f_xy']['p1.x'][0][0], 39.0, 1e-6)

        # Make sure step_size is used
        # Derivative should be way high with this.
        comp.fd_options['step_size'] = 1e5

        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        self.assertGreater(J['comp.f_xy']['p1.x'][0][0], 1000.0)

    def test_fd_options_step_size_precedence(self):

        class MyComp(Component):

            def __init__(self):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 3.0, step_size = 1e-6)

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 7.0*params['x1']**2 + 7.0*params['x2']**2

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p1.x2', 'comp.x2')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1.0e-4

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x1', 'p1.x2'], ['comp.y'], return_format='dict')
        x1_err = J['comp.y']['p1.x1'] - 42.0
        x2_err = J['comp.y']['p1.x2'] - 42.0

        assert_rel_error(self, x1_err, 7e-4, 1e-1)
        assert_rel_error(self, x2_err, 7e-6, 1e-1)

    def test_fd_options_step_size_ambiguous(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', FDpropsComp(step_size=1.e-6), promotes=['x2'])
        comp2 = prob.root.add('comp2', FDpropsComp(step_size=1.001e-6), promotes=['x2'])
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1.0e-4

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("The following parameters have the same promoted name, "
                             "'x2', but different 'step_size' values: [('comp.x2', 1e-06), "
                             "('comp2.x2', 1.001e-06)]" in str(err))

    def test_fd_options_step_type_ambiguous(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', FDpropsComp(step_type='absolute'), promotes=['x2'])
        comp2 = prob.root.add('comp2', FDpropsComp(step_type='relative'), promotes=['x2'])
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1.0e-4

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("The following parameters have the same promoted name, "
                             "'x2', but different 'step_type' values: [('comp.x2', 'absolute'), "
                             "('comp2.x2', 'relative')]" in str(err))

    def test_fd_options_form_ambiguous(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', FDpropsComp(form='central'), promotes=['x2'])
        comp2 = prob.root.add('comp2', FDpropsComp(form='forward'), promotes=['x2'])
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1.0e-4

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("The following parameters have the same promoted name, "
                             "'x2', but different 'form' values: [('comp.x2', 'central'), "
                             "('comp2.x2', 'forward')]" in str(err))

    def test_fd_options_step_type_precedence(self):

        class MyComp(Component):

            def __init__(self):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 3.0, step_type = 'absolute')

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 7.0*params['x1']**2 + 7.0*params['x2']**2

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p1.x2', 'comp.x2')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_type'] = 'relative'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x1', 'p1.x2'], ['comp.y'], return_format='dict')
        x1_err = J['comp.y']['p1.x1'] - 42.0
        x2_err = J['comp.y']['p1.x2'] - 42.0

        assert_rel_error(self, x1_err, 2.1e-5, 1e-1)
        assert_rel_error(self, x2_err, 7e-6, 1e-1)

    def test_fd_options_form_precedence(self):

        class MyComp(Component):

            def __init__(self):
                super(MyComp, self).__init__()

                # Params
                self.add_param('x1', 3.0)
                self.add_param('x2', 3.0, form = 'central')

                # Unknowns
                self.add_output('y', 5.5)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Doesn't do much. """
                unknowns['y'] = 7.0*params['x1']**2 + 7.0*params['x2']**2

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', MyComp())
        prob.root.add('p1', IndepVarComp([('x1', 3.0), ('x2', 3.0)]))
        prob.root.connect('p1.x1', 'comp.x1')
        prob.root.connect('p1.x2', 'comp.x2')

        comp.fd_options['force_fd'] = True
        comp.fd_options['form'] = 'forward'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['p1.x1', 'p1.x2'], ['comp.y'], return_format='dict')
        x1_err = J['comp.y']['p1.x1'] - 42.0
        x2_err = J['comp.y']['p1.x2'] - 42.0

        assert_rel_error(self, x1_err, 7e-6, 1e-1)
        assert_rel_error(self, x2_err, 5.4e-10, 1e-1)

    def test_fd_options_form(self):

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', Paraboloid())
        prob.root.add('p1', IndepVarComp('x', 15.0))
        prob.root.add('p2', IndepVarComp('y', 15.0))
        prob.root.connect('p1.x', 'comp.x')
        prob.root.connect('p2.y', 'comp.y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['form'] = 'forward'

        indep_list = ['p1.x']
        unknowns_list = ['comp.f_xy']
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(indep_list, unknowns_list, return_format='dict')
        assert_rel_error(self, J['comp.f_xy']['p1.x'][0][0], 39.0, 1e-6)

        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        # Make sure it gives good result with small stepsize
        comp.fd_options['form'] = 'backward'

        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        assert_rel_error(self, J['comp.f_xy']['p1.x'][0][0], 39.0, 1e-6)

        # Make sure it gives good result with small stepsize
        comp.fd_options['form'] = 'central'

        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        assert_rel_error(self, J['comp.f_xy']['p1.x'][0][0], 39.0, 1e-6)

        # Now, Make sure we really are going foward and backward
        comp.fd_options['form'] = 'forward'
        comp.fd_options['step_size'] = 1e3
        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        self.assertGreater(J['comp.f_xy']['p1.x'][0][0], 0.0)

        comp.fd_options['form'] = 'backward'
        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        self.assertLess(J['comp.f_xy']['p1.x'][0][0], 0.0)

        # Central should get pretty close even for the bad stepsize
        comp.fd_options['form'] = 'central'
        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        assert_rel_error(self, J['comp.f_xy']['p1.x'][0][0], 39.0, 1e-1)

    def test_fd_options_step_type(self):

        class ScaledParaboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(ScaledParaboloid, self).__init__()

                # Params
                self.add_param('x', 1.0)
                self.add_param('y', 1.0)

                # Unknowns
                self.add_output('f_xy', 0.0)

                self.scale = 1.0e-6

            def solve_nonlinear(self, params, unknowns, resids):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Optimal solution (minimum): x = 6.6667; y = -7.3333
                """

                x = params['x']
                y = params['y']

                f_xy = ((x-3.0)**2 + x*y + (y+4.0)**2 - 3.0)
                unknowns['f_xy'] = self.scale*f_xy

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives"""

                x = params['x']
                y = params['y']
                J = {}

                J['f_xy', 'x'] = (2.0*x - 6.0 + y) * self.scale
                J['f_xy', 'y'] = (2.0*y + 8.0 + x) * self.scale

                return J

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', ScaledParaboloid())
        prob.root.add('p1', IndepVarComp('x', 8.0*comp.scale))
        prob.root.add('p2', IndepVarComp('y', 8.0*comp.scale))
        prob.root.connect('p1.x', 'comp.x')
        prob.root.connect('p2.y', 'comp.y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_type'] = 'absolute'

        prob.setup(check=False)
        prob.run()

        J1 = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')

        comp.fd_options['step_type'] = 'relative'
        J2 = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')

        # Couldnt put together a case where one is much worse, so just make sure they
        # are not equal.
        self.assertNotEqual(self, J1['comp.f_xy']['p1.x'][0][0],
                                  J2['comp.f_xy']['p1.x'][0][0])

    def test_fd_options_meta_step_size(self):

        class MetaParaboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(MetaParaboloid, self).__init__()

                # Params
                self.add_param('x', 1.0, step_size = 1.0e5)
                self.add_param('y', 1.0, step_size = 1.0e5)

                # Unknowns
                self.add_output('f_xy', 0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Optimal solution (minimum): x = 6.6667; y = -7.3333
                """

                x = params['x']
                y = params['y']

                f_xy = ((x-3.0)**2 + x*y + (y+4.0)**2 - 3.0)
                unknowns['f_xy'] = f_xy

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives"""

                x = params['x']
                y = params['y']
                J = {}

                J['f_xy', 'x'] = (2.0*x - 6.0 + y)
                J['f_xy', 'y'] = (2.0*y + 8.0 + x)

                return J

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', MetaParaboloid())
        prob.root.add('p1', IndepVarComp('x', 15.0))
        prob.root.add('p2', IndepVarComp('y', 15.0))
        prob.root.connect('p1.x', 'comp.x')
        prob.root.connect('p2.y', 'comp.y')

        comp.fd_options['force_fd'] = True

        prob.setup(check=False)
        prob.run()

        # Make sure bad meta step_size is used
        # Derivative should be way high with this.

        J = prob.calc_gradient(['p1.x'], ['comp.f_xy'], return_format='dict')
        self.assertGreater(J['comp.f_xy']['p1.x'][0][0], 1000.0)

    def test_fd_options_meta_form(self):

        class MetaParaboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(MetaParaboloid, self).__init__()

                # Params
                self.add_param('x1', 1.0, form = 'forward')
                self.add_param('x2', 1.0, form = 'backward')
                self.add_param('y', 1.0)

                # Unknowns
                self.add_output('f_xy', 0.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                Optimal solution (minimum): x = 6.6667; y = -7.3333
                """

                x1 = params['x1']
                x2 = params['x2']
                y = params['y']

                f_xy = ((x1-3.0)**2 + (x2-3.0)**2 + (x2+x2)*y + (y+4.0)**2 - 3.0)
                unknowns['f_xy'] = f_xy

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives"""

                x1 = params['x1']
                x2 = params['x2']
                y = params['y']
                J = {}

                J['f_xy', 'x1'] = (2.0*x1 - 6.0 + x2*y)
                J['f_xy', 'x2'] = (2.0*x2 - 6.0 + x1*y)
                J['f_xy', 'y'] = (2.0*y + 8.0 + x1 + x2)

                return J

        prob = Problem()
        prob.root = Group()
        comp = prob.root.add('comp', MetaParaboloid())
        prob.root.add('p11', IndepVarComp('x1', 15.0))
        prob.root.add('p12', IndepVarComp('x2', 15.0))
        prob.root.add('p2', IndepVarComp('y', 15.0))
        prob.root.connect('p11.x1', 'comp.x1')
        prob.root.connect('p12.x2', 'comp.x2')
        prob.root.connect('p2.y', 'comp.y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1e3

        params_list = ['p11.x1']
        unknowns_list = ['comp.f_xy']

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(params_list, unknowns_list, return_format='dict')
        self.assertGreater(J['comp.f_xy']['p11.x1'][0][0], 0.0)

        J = prob.calc_gradient(['p12.x2'], unknowns_list, return_format='dict')
        self.assertLess(J['comp.f_xy']['p12.x2'][0][0], 0.0)


if __name__ == "__main__":
    unittest.main()

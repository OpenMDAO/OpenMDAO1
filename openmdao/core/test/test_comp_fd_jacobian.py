""" Unit test for the fd_jacobian method on Component. This method peforms a
finite difference."""

from __future__ import print_function
from collections import OrderedDict
import unittest

import numpy as np

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.system import System
from openmdao.core.vecwrapper import SrcVecWrapper

from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.test.simplecomps import SimpleArrayComp, \
                                      SimpleImplicitComp, \
                                      Paraboloid
from openmdao.test.testutil import assert_equal_jacobian, assert_rel_error


class TestProb(Problem):

    def __init__(self):
        super(TestProb, self).__init__()

        self.root = root = Group()
        root.add('c1', SimpleArrayComp())
        root.add('p1', ParamComp('p', 1*np.ones(2)))
        root.connect('p1:p','c1:x')

        root.add('ci1', SimpleImplicitComp())
        root.add('pi1', ParamComp('p', 1.))
        root.connect('pi1:p','ci1:x')


class CompFDTestCase(unittest.TestCase):
    """ Some basic tests of the fd_jacobian method in Component."""

    def setUp(self):
        self.p = TestProb()
        self.p.setup()

    def test_correct_keys_in_jac(self):

        expected_keys=[('y', 'x')]

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2)),
                             'pathname' : 'x',
                             'relative_name' : 'x',
                             'shape': 2, 'size' : 2}

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2)),
                               'pathname' : 'y',
                               'relative_name' : 'y',
                             'shape': 2, 'size' : 2 }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2)),
                             'pathname' : 'y',
                             'relative_name' : 'y',
                             'shape': 2, 'size' : 2}

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

    def test_correct_vals_in_jac(self):

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2)),
                             'pathname' : 'x',
                             'relative_name' : 'x',
                             'shape': 2, 'size' : 2 }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2)),
                               'pathname' : 'y',
                               'relative_name' : 'y',
                             'shape': 2, 'size' : 2 }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2)),
                             'pathname' : 'y',
                             'relative_name' : 'y',
                             'shape': 2, 'size' : 2 }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        self.p.subsystem('c1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2.,  7.],[ 5., -3.]])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

        #Got lucky that the way this comp was written, it would accept any square
        # matrix. But provided jacobian would be really wrong!

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.ones((2, 2)),
                             'pathname' : 'x',
                             'relative_name' : 'x',
                             'shape': (2,2), 'size' : 4 }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.zeros((2, 2)),
                               'pathname' : 'y',
                               'relative_name' : 'y',
                             'shape': (2,2), 'size' : 4 }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.zeros((2, 2)),
                             'pathname' : 'y',
                             'relative_name' : 'y',
                             'shape': (2,2), 'size' : 4 }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        self.p.subsystem('c1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('c1').fd_jacobian(params, unknowns, resids)

        expected_jac = {('y', 'x'): np.array([[ 2,  0,  7,  0],
                                              [ 0,  2,  0,  7],
                                              [ 5,  0, -3,  0],
                                              [ 0,  5,  0, -3]
                                             ])}

        assert_equal_jacobian(self, jac, expected_jac, 1e-8)

    def test_correct_vals_in_jac_implicit(self):

        params_dict = OrderedDict()
        params_dict['x'] = { 'val': np.array([0.5]),
                             'pathname' : 'x',
                             'relative_name' : 'x',
                             'shape': (1,), 'size' : 1 }

        unknowns_dict = OrderedDict()
        unknowns_dict['y'] = { 'val': np.array([0.0]),
                               'pathname' : 'y',
                               'relative_name' : 'y',
                             'shape': (1,), 'size' : 1 }
        unknowns_dict['z'] = { 'val': np.array([0.0]),
                               'pathname' : 'z',
                               'relative_name' : 'z',
                             'shape': (1,), 'size' : 1 }

        resids_dict = OrderedDict()
        resids_dict['y'] = { 'val': np.array([0.0]),
                             'pathname' : 'y',
                             'relative_name' : 'y',
                             'shape': (1,), 'size' : 1 }
        resids_dict['z'] = { 'val': np.array([0.0]),
                             'pathname' : 'z',
                             'relative_name' : 'z',
                             'shape': (1,), 'size' : 1 }

        params = SrcVecWrapper()
        params.setup(params_dict, store_byobjs=True)

        unknowns = SrcVecWrapper()
        unknowns.setup(unknowns_dict, store_byobjs=True)

        resids = SrcVecWrapper()
        resids.setup(resids_dict, store_byobjs=True)

        # Partials

        self.p.subsystem('ci1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('ci1').fd_jacobian(params, unknowns, resids)
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
        self.p.subsystem('ci1').atol = 1e-14
        self.p.subsystem('ci1').solve_nonlinear(params, unknowns, resids)

        jac = self.p.subsystem('ci1').fd_jacobian(params, unknowns, resids, total_derivs=True)
        expected_jac = {}
        expected_jac[('y', 'x')] = -2.5555555555555554
        expected_jac[('z', 'x')] = -1.7777777777777777

        assert_equal_jacobian(self, jac, expected_jac, 1e-5)


class CompFDinSystemTestCase(unittest.TestCase):
    """ Tests automatic finite difference of a component in a full problem."""

    def test_no_derivatives(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', ExecComp('y=x*2.0'))
        top.root.add('p1', ParamComp('x', 2.0))
        top.root.connect('p1:x', 'comp:x')

        comp.fd_options['force_fd'] = True

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x'], ['comp:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp:y']['comp:x'][0][0], 2.0, 1e-6)

        J = top.calc_gradient(['comp:x'], ['comp:y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['comp:y']['comp:x'][0][0], 2.0, 1e-6)

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


            def jacobian(self, params, unknowns, resids):
                """Never Call."""
                raise RuntimeError("This should have been overridden by force_fd.")


        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', OverrideComp())
        top.root.add('p1', ParamComp('x', 2.0))
        top.root.connect('p1:x', 'comp:x')

        comp.fd_options['force_fd'] = True

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x'], ['comp:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp:y']['comp:x'][0][0], 7.0, 1e-6)

    def test_fd_options_step_size(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', Paraboloid())
        top.root.add('p1', ParamComp('x', 15.0))
        top.root.add('p2', ParamComp('y', 15.0))
        top.root.connect('p1:x', 'comp:x')
        top.root.connect('p2:y', 'comp:y')

        comp.fd_options['force_fd'] = True

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        assert_rel_error(self, J['comp:f_xy']['comp:x'][0][0], 39.0, 1e-6)

        # Make sure step_size is used
        # Derivative should be way high with this.
        comp.fd_options['step_size'] = 1e5

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        self.assertGreater(J['comp:f_xy']['comp:x'][0][0], 1000.0)

    def test_fd_options_form(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', Paraboloid())
        top.root.add('p1', ParamComp('x', 15.0))
        top.root.add('p2', ParamComp('y', 15.0))
        top.root.connect('p1:x', 'comp:x')
        top.root.connect('p2:y', 'comp:y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['form'] = 'forward'

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        assert_rel_error(self, J['comp:f_xy']['comp:x'][0][0], 39.0, 1e-6)

        # Make sure it gives good result with small stepsize
        comp.fd_options['form'] = 'backward'

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        assert_rel_error(self, J['comp:f_xy']['comp:x'][0][0], 39.0, 1e-6)

        # Make sure it gives good result with small stepsize
        comp.fd_options['form'] = 'central'

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        assert_rel_error(self, J['comp:f_xy']['comp:x'][0][0], 39.0, 1e-6)

        # Now, Make sure we really are going foward and backward
        comp.fd_options['form'] = 'forward'
        comp.fd_options['step_size'] = 1e3
        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        self.assertGreater(J['comp:f_xy']['comp:x'][0][0], 0.0)

        comp.fd_options['form'] = 'backward'
        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        self.assertLess(J['comp:f_xy']['comp:x'][0][0], 0.0)

        # Central should get pretty close even for the bad stepsize
        comp.fd_options['form'] = 'central'
        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        assert_rel_error(self, J['comp:f_xy']['comp:x'][0][0], 39.0, 1e-1)

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

            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives"""

                x = params['x']
                y = params['y']
                J = {}

                J['f_xy', 'x'] = (2.0*x - 6.0 + y) * self.scale
                J['f_xy', 'y'] = (2.0*y + 8.0 + x) * self.scale

                return J

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', ScaledParaboloid())
        top.root.add('p1', ParamComp('x', 8.0*comp.scale))
        top.root.add('p2', ParamComp('y', 8.0*comp.scale))
        top.root.connect('p1:x', 'comp:x')
        top.root.connect('p2:y', 'comp:y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_type'] = 'absolute'

        top.setup()
        top.run()

        J1 = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')

        comp.fd_options['step_type'] = 'relative'
        J2 = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')

        # Couldnt put together a case where one is much worse, so just make sure they
        # are not equal.
        self.assertNotEqual(self, J1['comp:f_xy']['comp:x'][0][0],
                                  J2['comp:f_xy']['comp:x'][0][0])

    def test_fd_options_meta_step_size(self):

        class MetaParaboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(MetaParaboloid, self).__init__()

                # Params
                self.add_param('x', 1.0, fd_step_size = 1.0e5)
                self.add_param('y', 1.0, fd_step_size = 1.0e5)

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

            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives"""

                x = params['x']
                y = params['y']
                J = {}

                J['f_xy', 'x'] = (2.0*x - 6.0 + y)
                J['f_xy', 'y'] = (2.0*y + 8.0 + x)

                return J

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', MetaParaboloid())
        top.root.add('p1', ParamComp('x', 15.0))
        top.root.add('p2', ParamComp('y', 15.0))
        top.root.connect('p1:x', 'comp:x')
        top.root.connect('p2:y', 'comp:y')

        comp.fd_options['force_fd'] = True

        top.setup()
        top.run()

        # Make sure bad meta step_size is used
        # Derivative should be way high with this.

        J = top.calc_gradient(['comp:x'], ['comp:f_xy'], return_format='dict')
        self.assertGreater(J['comp:f_xy']['comp:x'][0][0], 1000.0)

    def test_fd_options_meta_form(self):

        class MetaParaboloid(Component):
            """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

            def __init__(self):
                super(MetaParaboloid, self).__init__()

                # Params
                self.add_param('x1', 1.0, fd_form = 'forward')
                self.add_param('x2', 1.0, fd_form = 'backward')
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

            def jacobian(self, params, unknowns, resids):
                """Analytical derivatives"""

                x1 = params['x1']
                x2 = params['x2']
                y = params['y']
                J = {}

                J['f_xy', 'x1'] = (2.0*x1 - 6.0 + x2*y)
                J['f_xy', 'x2'] = (2.0*x2 - 6.0 + x1*y)
                J['f_xy', 'y'] = (2.0*y + 8.0 + x1 + x2)

                return J

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', MetaParaboloid())
        top.root.add('p11', ParamComp('x1', 15.0))
        top.root.add('p12', ParamComp('x2', 15.0))
        top.root.add('p2', ParamComp('y', 15.0))
        top.root.connect('p11:x1', 'comp:x1')
        top.root.connect('p12:x2', 'comp:x2')
        top.root.connect('p2:y', 'comp:y')

        comp.fd_options['force_fd'] = True
        comp.fd_options['step_size'] = 1e3

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x1'], ['comp:f_xy'], return_format='dict')
        self.assertGreater(J['comp:f_xy']['comp:x1'][0][0], 0.0)

        J = top.calc_gradient(['comp:x2'], ['comp:f_xy'], return_format='dict')
        self.assertLess(J['comp:f_xy']['comp:x2'][0][0], 0.0)


if __name__ == "__main__":
    unittest.main()

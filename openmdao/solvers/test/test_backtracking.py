""" Test for the Backktracking Line Search"""

import unittest
from math import exp

import numpy as np

from openmdao.api import Problem, Group, NonLinearSolver, IndepVarComp, \
                         Component, AnalysisError
from openmdao.solvers.backtracking import BackTracking
from openmdao.solvers.newton import Newton
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.util import assert_rel_error


class TrickyComp(Component):

    def __init__(self):
        super(TrickyComp, self).__init__()

        # Inputs
        self.add_param('y', 1.2278849186466743)

        # States
        self.add_state('x', val=1.0)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        y = params['y']
        x = unknowns['x']

        resids['x'] = 0.5*x*x + 2.0*x + exp(-16.0*x*x) + 2.0*exp(-5.0*x) - y
        #print('x', x, 'res', resids['x'])

    def solve_nonlinear(self, params, unknowns, resids):
        """ This is a dummy comp that doesn't modify its state."""
        pass

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        x = unknowns['x']

        J = {}

        # State equation
        J[('x', 'y')] = -1.0
        J[('x', 'x')] = x + 2.0 - 32.0*x*exp(-16.0*x*x) - 10.0*exp(-5.0*x)

        return J


class TestBackTracking(unittest.TestCase):

    def test_newton_with_backtracking(self):

        top = Problem()
        root = top.root = Group()
        root.add('comp', TrickyComp())
        root.add('p', IndepVarComp('y', 1.2278849186466743))
        root.connect('p.y', 'comp.y')

        root.nl_solver = Newton()
        root.ln_solver = ScipyGMRES()
        root.nl_solver.line_search = BackTracking()
        root.nl_solver.line_search.options['maxiter'] = 100
        root.nl_solver.line_search.options['c'] = 0.5
        root.nl_solver.options['alpha'] = 10.0

        top.setup(check=False)
        top['comp.x'] = 1.0
        top.print_all_convergence(level=1)
        top.run()

        assert_rel_error(self, top['comp.x'], .3968459, .0001)

    def test_newton_with_backtracking_analysis_error(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.root.nl_solver.line_search = BackTracking()
        top.root.nl_solver.line_search.options['maxiter'] = 2
        top.root.nl_solver.line_search.options['err_on_maxiter'] = True
        top.root.nl_solver.line_search.options['c'] = 1.0
        top.root.nl_solver.options['alpha'] = 10.0

        top.setup(check=False)

        try:
            top.run()
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': BackTracking failed to converge after 2 iterations.")
        else:
            self.fail("AnalysisError expected")

    def test_bounds_backtracking(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 2.0, lower=1.5, upper=2.5)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 2.0
        top.run()

        self.assertEqual(top['comp.z'], 1.5)

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 0.5
        top.run()

        self.assertEqual(top['comp.z'], 2.5)

    def test_bounds_backtracking_start_violated_high(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 3.0, lower=1.5, upper=2.5)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 2.0
        top.run()

        self.assertEqual(top['comp.z'], 2.5)

    def test_bounds_backtracking_start_violated_low(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 1.2, lower=1.5, upper=2.5)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 2.0
        top.run()

        self.assertEqual(top['comp.z'], 1.5)

    def test_bounds_backtracking_lower_only(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 2.0, lower=1.5)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 2.0
        top.run()

        self.assertEqual(top['comp.z'], 1.5)

    def test_bounds_backtracking_just_upper(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 2.0, upper=2.5)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.array([1.0])
                J[('y', 'z')] = np.array([2.0])

                # State equation
                J[('z', 'z')] = np.array([params['x'] + 1.0])
                J[('z', 'x')] = np.array([unknowns['z']])

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', 1.0))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 0.5
        top.run()

        self.assertEqual(top['comp.z'], 2.5)

    def test_bounds_backtracking_arrays_vector_alpha(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', np.zeros((3, 1)))

                # Unknowns
                self.add_output('y', np.zeros((3, 1)))

                # States
                self.add_state('z', 2.0*np.ones((3, 1)), lower=1.5, upper=np.array([[2.6, 2.5, 2.65]]).T)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = x*z + z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x + 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.diag(np.array([1.0, 1.0, 1.0]))
                J[('y', 'z')] = np.diag(np.array([2.0, 2.0, 2.0]))

                # State equation
                J[('z', 'z')] = (params['x'] + 1.0) * np.eye(3)
                J[('z', 'x')] = unknowns['z'] * np.eye(3)

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = np.array([2.0, 2.0, 2.0])
        top.run()

        self.assertEqual(top['comp.z'][0], 1.5)
        self.assertEqual(top['comp.z'][1], 1.5)
        self.assertEqual(top['comp.z'][2], 1.5)

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 6
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 0.5*np.ones((3, 1))
        top.run()

        # Each bound is observed
        self.assertEqual(top['comp.z'][0], 2.6)
        self.assertEqual(top['comp.z'][1], 2.5)
        self.assertEqual(top['comp.z'][2], 2.65)

    def test_bounds_backtracking_arrays_lower_vector_alpha(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', np.zeros((3, 1)))

                # Unknowns
                self.add_output('y', np.zeros((3, 1)))

                # States
                self.add_state('z', -2.0*np.ones((3, 1)), upper=-1.5, lower=np.array([[-2.6, -2.5, -2.65]]).T)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = -x*z - z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x - 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.diag(np.array([1.0, 1.0, 1.0]))
                J[('y', 'z')] = -np.diag(np.array([2.0, 2.0, 2.0]))

                # State equation
                J[('z', 'z')] = -(params['x'] + 1.0) * np.eye(3)
                J[('z', 'x')] = -unknowns['z'] * np.eye(3)

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = np.array([2.0, 2.0, 2.0])
        top.run()

        self.assertEqual(top['comp.z'][0], -1.5)
        self.assertEqual(top['comp.z'][1], -1.5)
        self.assertEqual(top['comp.z'][2], -1.5)

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 0.5*np.ones((3, 1))
        top.run()

        # Each bound is observed
        self.assertEqual(top['comp.z'][0], -2.6)
        self.assertEqual(top['comp.z'][1], -2.5)
        self.assertEqual(top['comp.z'][2], -2.65)

    def test_bounds_array_initially_violated(self):

        class SimpleImplicitComp(Component):
            """ A Simple Implicit Component with an additional output equation.

            f(x,z) = xz + z - 4
            y = x + 2z

            Sol: when x = 0.5, z = 2.666
            Sol: when x = 2.0, z = 1.333

            Coupled derivs:

            y = x + 8/(x+1)
            dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

            z = 4/(x+1)
            dz_dx = -4/(x+1)**2 = -1.7777777777777777
            """

            def __init__(self):
                super(SimpleImplicitComp, self).__init__()

                # Params
                self.add_param('x', np.zeros((3, 1)))

                # Unknowns
                self.add_output('y', np.zeros((3, 1)))
                self.add_output('z_int', 5, upper=6)

                # States
                self.add_state('z', np.array([[-1.0, -2.0, -2.8]]).T, upper=-1.5, lower=np.array([[-2.6, -2.5, -2.65]]).T)

                self.maxiter = 10
                self.atol = 1.0e-12

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']
                resids['z'] = -x*z - z - 4.0

                # Output equations need to evaluate a residual just like an explicit comp.
                resids['y'] = x - 2.0*z - unknowns['y']

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""

                J = {}

                # Output equation
                J[('y', 'x')] = np.diag(np.array([1.0, 1.0, 1.0]))
                J[('y', 'z')] = -np.diag(np.array([2.0, 2.0, 2.0]))

                # State equation
                J[('z', 'z')] = -(params['x'] + 1.0) * np.eye(3)
                J[('z', 'x')] = -unknowns['z'] * np.eye(3)

                return J

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem()
        top.root = Group()
        top.root.add('comp', SimpleImplicitComp())
        top.root.ln_solver = ScipyGMRES()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = np.array([-2.0, -2.0, -2.0])
        top.run()

        self.assertEqual(top['comp.z'][0], -1.5)
        self.assertEqual(top['comp.z'][1], -1.5)
        self.assertEqual(top['comp.z'][2], -2.65)


if __name__ == "__main__":
    unittest.main()

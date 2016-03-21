""" Unit testing for the Backtracking sub-solver. """

import unittest

import numpy as np

from openmdao.api import Problem, Group, NonLinearSolver, IndepVarComp, Component, AnalysisError
from openmdao.solvers.backtracking import BackTracking
from openmdao.solvers.newton import Newton
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.util import assert_rel_error


class FakeSolver(NonLinearSolver):
    """ Does nothing but invoke a line search."""

    def __init__(self):
        super(FakeSolver, self).__init__()
        self.sub = BackTracking()

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Calc deriv then do line search."""

        # Perform an initial run to propagate srcs to targets.
        system.children_solve_nonlinear(None)
        system.apply_nonlinear(params, unknowns, resids)

        # Linearize Model with partial derivatives
        system._sys_linearize(params, unknowns, resids, total_derivs=False)

        # Calculate direction to take step
        arg = system.drmat[None]
        result = system.dumat[None]

        # Step waaaaay to far so we have to backtrack
        arg.vec[:] = resids.vec*100
        system.solve_linear(system.dumat, system.drmat, [None], mode='fwd')

        unknowns.vec += result.vec

        self.sub.solve(params, unknowns, resids, system, self, 1.0, 1.0, 1.0)

class TestBackTracking(unittest.TestCase):

    def test_newton_with_backtracking(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.root.nl_solver.line_search.options['atol'] = 1e-12
        top.root.nl_solver.line_search.options['rtol'] = 1e-12
        top.root.nl_solver.line_search.options['maxiter'] = 3

        # This is a very contrived test, but we step 8 times farther than we
        # should, then allow the line search to backtrack 3 steps, which
        # takes us back to 1.0.
        top.root.nl_solver.options['alpha'] = 8.0

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['state_eq.y2_command'], 12.05848819, .00001)

    def test_newton_with_backtracking_analysis_error(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.root.nl_solver.line_search.options['atol'] = 1e-12
        top.root.nl_solver.line_search.options['rtol'] = 1e-12
        top.root.nl_solver.line_search.options['maxiter'] = 2
        top.root.nl_solver.line_search.options['err_on_maxiter'] = True

        # This is a very contrived test, but we step 8 times farther than we
        # should, then allow the line search to backtrack 3 steps, which
        # takes us back to 1.0.
        top.root.nl_solver.options['alpha'] = 8.0

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


    def test_bounds_backtracking_arrays(self):

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
                self.add_state('z', 2.0*np.ones((3, 1)), lower=1.5, upper=np.array([2.5, 2.6, 2.7]))

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
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.connect('px.x', 'comp.x')
        top.setup(check=False)

        top['px.x'] = 0.5*np.ones((3, 1))
        top.run()

        # Most restrictive bound is observed
        self.assertEqual(top['comp.z'][0], 2.5)
        self.assertEqual(top['comp.z'][1], 2.5)
        self.assertEqual(top['comp.z'][2], 2.5)

if __name__ == "__main__":
    unittest.main()

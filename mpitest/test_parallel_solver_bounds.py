""" Testing out that Newton/Backtracking can allgather bounds and step the correct amount."""

import os
import unittest
import numpy as np

from openmdao.api import Problem, Group, NonLinearSolver, IndepVarComp, Component, ParallelGroup
from openmdao.solvers.backtracking import BackTracking
from openmdao.solvers.newton import Newton
from openmdao.test.mpi_util import MPITestCase
from openmdao.core.mpi_wrap import MPI
from openmdao.test.util import assert_rel_error

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    from openmdao.solvers.petsc_ksp import PetscKSP as lin_solver
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    from openmdao.solvers.scipy_gmres import ScipyGMRES as lin_solver


class TestParallelBounds(MPITestCase):

    N_PROCS = 2

    def test_bounds_backtracking(self):

        class SimpleImplicitComp1(Component):
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
                super(SimpleImplicitComp1, self).__init__()

                # Params
                self.add_param('x', 0.5)

                # Unknowns
                self.add_output('y', 0.0)

                # States
                self.add_state('z', 2.0, lower=1.4, upper=2.5)

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

        class SimpleImplicitComp2(Component):
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
                super(SimpleImplicitComp2, self).__init__()

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

        top = Problem(impl=impl)
        top.root = Group()
        par = top.root.add('par', ParallelGroup())
        par.add('comp1', SimpleImplicitComp1())
        par.add('comp2', SimpleImplicitComp2())
        top.root.ln_solver = lin_solver()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px1', IndepVarComp('x', 1.0))
        top.root.add('px2', IndepVarComp('x', 1.0))

        # TODO: This is to get around a bug in checks. It should be fixed soon.
        par.ln_solver = lin_solver()

        top.root.connect('px1.x', 'par.comp1.x')
        top.root.connect('px2.x', 'par.comp2.x')
        top.setup(check=False)

        top['px1.x'] = 2.0
        top['px2.x'] = 2.0
        top.run()

        # Comp2 has a tighter lower bound. This test makes sure that we
        # allgathered and used the lowest alpha.

        if top.root.par.comp1.is_active():
            self.assertEqual(top['par.comp1.z'], 1.5)
        if top.root.par.comp2.is_active():
            self.assertEqual(top['par.comp2.z'], 1.5)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

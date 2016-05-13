""" Test out newton solve with the array alpha on the backtracking linesearch."""

from __future__ import print_function

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, Component, ParallelGroup, \
                         Newton
from openmdao.test.mpi_util import MPITestCase

try:
    from mpi4py import MPI
    from openmdao.core.petsc_impl import PetscImpl as impl
    from openmdao.solvers.petsc_ksp import PetscKSP
except ImportError:
    impl = None


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

class TestNewtonBacktrackingMPI(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if impl is None:
            raise unittest.SkipTest("Can't run this test (even in serial) without mpi4py and petsc4py")

    def test_newton_backtrack_MPI(self):

        #------------------------------------------------------
        # Test that Newton doesn't drive it past lower bounds
        #------------------------------------------------------

        top = Problem(impl=impl)
        top.root = Group()
        par = top.root.add('par', ParallelGroup())
        par.add('comp1', SimpleImplicitComp1())
        par.add('comp2', SimpleImplicitComp2())
        top.root.ln_solver = PetscKSP()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.nl_solver.line_search.options['vector_alpha'] = True

        top.root.connect('px.x', 'par.comp1.x')
        top.root.connect('px.x', 'par.comp2.x')
        top.setup(check=False)

        top['px.x'] = np.array([2.0, 2.0, 2.0])
        top.run()

        if not MPI or self.comm.rank == 0:
            self.assertEqual(top['par.comp1.z'][0], 1.5)
            self.assertEqual(top['par.comp1.z'][1], 1.5)
            self.assertEqual(top['par.comp1.z'][2], 1.5)

        if not MPI or self.comm.rank == 1:
            self.assertEqual(top['par.comp2.z'][0], -1.5)
            self.assertEqual(top['par.comp2.z'][1], -1.5)
            self.assertEqual(top['par.comp2.z'][2], -1.5)

        #------------------------------------------------------
        # Test that Newton doesn't drive it past upper bounds
        #------------------------------------------------------

        top = Problem(impl=impl)
        top.root = Group()
        par = top.root.add('par', ParallelGroup())
        par.add('comp1', SimpleImplicitComp1())
        par.add('comp2', SimpleImplicitComp2())
        top.root.ln_solver = PetscKSP()
        top.root.nl_solver = Newton()
        top.root.nl_solver.options['maxiter'] = 5
        top.root.add('px', IndepVarComp('x', np.ones((3, 1))))

        top.root.nl_solver.line_search.options['vector_alpha'] = True

        top.root.connect('px.x', 'par.comp1.x')
        top.root.connect('px.x', 'par.comp2.x')
        top.setup(check=False)

        top['px.x'] = 0.5*np.ones((3, 1))
        top.run()

        # Each bound is observed
        if not MPI or self.comm.rank == 0:
            self.assertEqual(top['par.comp1.z'][0], 2.6)
            self.assertEqual(top['par.comp1.z'][1], 2.5)
            self.assertEqual(top['par.comp1.z'][2], 2.65)

        if not MPI or self.comm.rank == 1:
            self.assertEqual(top['par.comp2.z'][0], -2.6)
            self.assertEqual(top['par.comp2.z'][1], -2.5)
            self.assertEqual(top['par.comp2.z'][2], -2.65)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
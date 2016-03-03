""" Test from Andrew Ning's student, who found a bug in the Petsc KSP setup."""

from __future__ import print_function

import numpy as np

from openmdao.api import ExecComp, IndepVarComp, Group, NLGaussSeidel, \
                         Component, ParallelGroup, ScipyGMRES
from openmdao.api import Problem, ScipyOptimizer, pyOptSparseDriver
from openmdao.test.mpi_util import MPITestCase

try:
    from openmdao.solvers.petsc_ksp import PetscKSP
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

from openmdao.test.util import assert_rel_error


class SellarDis1(Component):
    """Component containing Discipline 1."""

    def __init__(self, problem_id=0):
        super(SellarDis1, self).__init__()

        self.problem_id = problem_id

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Local Design Variable
        self.add_param('x', val=0.)

        # Coupling parameter
        self.add_param('y2_%i' % problem_id, val=1.0)

        # Coupling output
        self.add_output('y1_%i' % problem_id, val=1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2"""

        problem_id = self.problem_id

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = params['y2_%i' % problem_id]

        unknowns['y1_%i' % problem_id] = z1**2 + z2 + x1 - 0.2*y2

    def linearize(self, params, unknowns, resids):
        """ Jacobian for Sellar discipline 1."""

        problem_id = self.problem_id

        J = {}

        J['y1_%i' % problem_id, 'y2_%i' % problem_id] = -0.2
        J['y1_%i' % problem_id, 'z'] = np.array([[2*params['z'][0], 1.0]])
        J['y1_%i' % problem_id, 'x'] = 1.0

        return J


class SellarDis2(Component):
    """Component containing Discipline 2."""

    def __init__(self, problem_id=0):
        super(SellarDis2, self).__init__()

        self.problem_id = problem_id

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Coupling parameter
        self.add_param('y1_%i' % problem_id, val=1.0)

        # Coupling output
        self.add_output('y2_%i' % problem_id, val=1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """Evaluates the equation
        y2 = y1**(.5) + z1 + z2"""

        problem_id = self.problem_id

        z1 = params['z'][0]
        z2 = params['z'][1]
        y1 = params['y1_%i' % problem_id]

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        y1 = abs(y1)

        unknowns['y2_%i' % problem_id] = y1**.5 + z1 + z2

    def linearize(self, params, unknowns, resids):
        """ Jacobian for Sellar discipline 2."""

        problem_id = self.problem_id

        J = {}

        J['y2_%i' % problem_id, 'y1_%i' % problem_id] = .5*params['y1_%i' % problem_id]**-.5
        J['y2_%i' % problem_id, 'z'] = np.array([[1.0, 1.0]])

        return J


class SellarDerivativesSubGroup(Group):

    def __init__(self, problem_id=0):
        super(SellarDerivativesSubGroup, self).__init__()

        self.add('d1', SellarDis1(problem_id=problem_id), promotes=['*'])
        self.add('d2', SellarDis2(problem_id=problem_id), promotes=['*'])

        self.nl_solver = NLGaussSeidel()
        self.nl_solver.options['atol'] = 1.0e-12

        self.ln_solver = PetscKSP()
        # self.ln_solver = ScipyGMRES()


class SellarDerivativesSuperGroup(Group):

    def __init__(self, nProblems=0):

        super(SellarDerivativesSuperGroup, self).__init__()

        self.add('px', IndepVarComp('x', 1.0), promotes=['*'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['*'])

        pg = self.add('manySellars', ParallelGroup(), promotes=['*'])
        print(nProblems)
        for problem_id in np.arange(0, nProblems):
            pg.add('Sellar%i' % problem_id, SellarDerivativesSubGroup(problem_id=problem_id), promotes=['*'])

        self.add('obj_cmp', ExecComp('obj = (x**2 + z[1] + y1_0 + exp(-y2_0)) + (x**2 + z[1] + y1_1 + exp(-y2_1)) + '
                                     '(x**2 + z[1] + y1_2 + exp(-y2_2)) + (x**2 + z[1] + y1_3 + exp(-y2_3))',
                                     z=np.array([0.0, 0.0]), x=0.0,
                                     y1_0=0.0, y2_0=0.0,
                                     y1_1=0.0, y2_1=0.0,
                                     y1_2=0.0, y2_2=0.0,
                                     y1_3=0.0, y2_3=0.0),
                 promotes=['*'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1_0'), promotes=['*'])
        self.add('con_cmp2', ExecComp('con2 = y2_0 - 24.0'), promotes=['*'])



class MPITests2(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if impl is None:
            raise unittest.SkipTest("Can't run this test (even in serial) without mpi4py and petsc4py")

    def test_run(self):

        nProblems = 4
        top = Problem(impl=impl)
        top.root = SellarDerivativesSuperGroup(nProblems=nProblems)

        top.driver = ScipyOptimizer()
        top.driver = pyOptSparseDriver()
        top.driver.options['optimizer'] = 'SNOPT'
        top.driver.opt_settings['Verify level'] = 0
        top.driver.opt_settings['Print file'] = 'SNOPT_print_petsctest.out'
        top.driver.opt_settings['Summary file'] = 'SNOPT_summary_petsctest.out'
        top.driver.opt_settings['Major iterations limit'] = 1000

        top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        top.driver.add_desvar('x', lower=0.0, upper=10.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1', upper=0.0)
        top.driver.add_constraint('con2', upper=0.0)

        top.root.ln_solver.options['single_voi_relevance_reduction'] = True
        top.setup(check=True)

        # Setting initial values for design variables
        top['x'] = 1.0
        top['z'] = np.array([5.0, 2.0])

        top.run()

        if top.root.comm.rank == 0:
            assert_rel_error(self, top['z'][0], 1.977639, 1.0e-6)
            assert_rel_error(self, top['z'][1], 0.0, 1.0e-6)
            assert_rel_error(self, top['x'], 0.0, 1.0e-6)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

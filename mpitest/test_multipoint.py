"""Testing out differenct approaches to Multipoint optimization"""

import unittest

from openmdao.api import IndepVarComp, ExecComp, Component, \
    ParallelGroup, Problem, Group, CaseDriver, SubProblem

from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error, ConcurrentTestCaseMixin, \
                               set_pyoptsparse_opt

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    from openmdao.solvers.petsc_ksp import PetscKSP as lin_solver
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    from openmdao.solvers.scipy_gmres import ScipyGMRES as lin_solver

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class Parab2D(Component):
    """A 2D Parabola."""

    def __init__(self):
        super(Parab2D, self).__init__()

        # Params
        self.add_param('xroot', 0.0)
        self.add_param('x', 0.0)
        self.add_param('yroot', 0.0)
        self.add_param('y', 0.0)

        # Unknowns
        self.add_output('z', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ compute: z = (x - xroot)**2 + (y - yroot)**2 """
        unknowns['z'] = (params['x'] - params['xroot'])**2 + (params['y'] - params['yroot'])**2

    def linearize(self, params, unknowns, resids):
        """ derivatives """
        J = {}
        J['z', 'x'] = 2.0*params['x'] - 2.0*self.xroot
        J['z', 'y'] = 2.0*params['y'] - 2.0*self.yroot
        return J


class MultiPointCaseDriver(CaseDriver):
    """ runs a number of points defined by `constants` and sums the responses
    """
    def __init__(self, constants=[], num_par_doe=1, load_balance=False):
        super(MultiPointCaseDriver, self).__init__(num_par_doe=num_par_doe,
                                                   load_balance=load_balance)
        self.constants = constants
        for cons, val in constants[0]:
            self.add_desvar(cons, val)

    def run(self, problem):
        # create one case for each set of constants (which defines a point)
        self.cases = []
        for i in range(len(self.constants)):
            case = []
            for dv in self._desvars:
                case.append((dv, problem[dv]))
            self.cases.append(case)

        print '-------'
        for case in self.cases:
            print(case)

        # run the cases
        super(MultiPointCaseDriver, self).run(problem)

        # sum the responses and populate unknowns
        outputs = {}
        for responses, _, _ in self.get_all_responses():
            for name, val in responses:
                if name not in outputs:
                    outputs[name] = val
                else:
                    outputs[name] += val

        for name, val in outputs.iteritems():
            if name in self.root.unknowns:
                self.root.unknowns[name] = val


class TestParallel(MPITestCase, ConcurrentTestCaseMixin):

    N_PROCS = 2

    def test_parab_2d_mpt(self):
        prob = Problem(impl=impl)
        root = prob.root = Group()
        root.ln_solver = lin_solver()

        root.add('p', IndepVarComp([('x', 0.0), ('y1', 0.0), ('y2', 0.0)]))

        par = root.add('par', ParallelGroup())
        par.add('c1', ExecComp('z = (x-2.0)**2 + (y-3.0)**2'))
        par.add('c2', ExecComp('z = (x-3.0)**2 + (y-5.0)**2'))

        root.add('sumcomp', ExecComp('sum = z1+z2'))

        root.connect('p.x',  'par.c1.x')
        root.connect('p.y1', 'par.c1.y')

        root.connect('p.x',  'par.c2.x')
        root.connect('p.y2', 'par.c2.y')

        root.connect('par.c1.z', 'sumcomp.z1')
        root.connect('par.c2.z', 'sumcomp.z2')

        driver = prob.driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False
        driver.add_desvar('p.x',  lower=-100, upper=100)
        driver.add_desvar('p.y1', lower=-100, upper=100)
        driver.add_desvar('p.y2', lower=-100, upper=100)
        driver.add_objective('sumcomp.sum')

        prob.setup(check=False)
        prob.run()

        if not MPI or self.comm.rank == 0:
            print("sum:", prob['sumcomp.sum'])
            print(prob['par.c1.x'], prob['par.c1.y'])
            assert_rel_error(self, prob['par.c1.x'], 2.5, 1.e-6)
            assert_rel_error(self, prob['par.c1.y'], 3.0, 1.e-6)

        if not MPI or self.comm.rank == 1:
            print(prob['par.c2.x'], prob['par.c2.y'])
            assert_rel_error(self, prob['par.c2.x'], 2.5, 1.e-6)
            assert_rel_error(self, prob['par.c2.y'], 5.0, 1.e-6)


class TestCaseDriver(MPITestCase, ConcurrentTestCaseMixin):

    N_PROCS = 2

    def test_parab_2d_mpt(self):
        # create multipoint problem for 2D parabola
        mpt = Problem(impl=impl)
        root = mpt.root = Group()

        root.deriv_options['type'] = 'fd'

        root.add('p', IndepVarComp([('xroot', 2.0), ('yroot', 3.0), ('x', 0.0), ('y', 0.0)]))
        root.add('c', Parab2D())

        root.connect('p.xroot', 'c.xroot')
        root.connect('p.yroot', 'c.yroot')

        root.connect('p.x', 'c.x')
        root.connect('p.y', 'c.y')

        # the two "points" are defined by these constants
        # the remaining independent vars will be the design vars
        constants = [
            [('p.xroot', 2.0), ('p.yroot', 3.0)],
            [('p.xroot', 3.0), ('p.yroot', 5.0)],
        ]

        mpt.driver = MultiPointCaseDriver(constants=constants, num_par_doe=2)
        mpt.driver.add_desvar('p.x')
        mpt.driver.add_desvar('p.y')
        mpt.driver.add_response('c.z')

        # create optimization problem
        prob = Problem(impl=impl)
        root = prob.root = Group()

        root.deriv_options['type'] = 'fd'

        root.add('p', IndepVarComp([('x', 0.0), ('y', 0.0)]))
        root.add('mpt', SubProblem(mpt, params=['p.x', 'p.y'], unknowns=['c.z']))

        root.connect('p.x', 'mpt.p.x')
        root.connect('p.y', 'mpt.p.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('p.x', lower=-100, upper=100)
        prob.driver.add_desvar('p.y', lower=-100, upper=100)
        prob.driver.add_objective('mpt.c.z')

        prob.setup(check=True)
        prob.run()

        if not MPI or self.comm.rank == 0:
            print('z =', prob['mpt.c.z'])
            print('at x =', prob['p.x'], 'y =', prob['p.y'])
            print('at x =', prob['mpt.p.x'], 'y =', prob['mpt.p.y'])


if __name__ == '__main__':
    if MPI:
        from openmdao.test.mpi_util import mpirun_tests
        mpirun_tests()
    else:
        unittest.main()

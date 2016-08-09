""" Testing out MPI optimization with pyopt_sparse"""

import unittest
import numpy as np

from openmdao.api import IndepVarComp, ExecComp, LinearGaussSeidel, Component, \
    ParallelGroup, Problem, Group, CaseDriver, SqliteRecorder, SubProblem

from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error, ConcurrentTestCaseMixin, \
                               set_pyoptsparse_opt
from openmdao.test.simple_comps import SimpleArrayComp
from openmdao.test.exec_comp_for_test import ExecComp4Test

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


class Parab1D(Component):
    """Just a 1D Parabola."""

    def __init__(self, root=1.0):
        super(Parab1D, self).__init__()

        self.root = root

        # Params
        self.add_param('x', 0.0)

        # Unknowns
        self.add_output('y', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ compute: y = (x - root)**2 + 7 """
        unknowns['y'] = (params['x'] - self.root)**2 + 7.0

    def linearize(self, params, unknowns, resids):
        """ derivs """
        J = {}
        J['y', 'x'] = 2.0*params['x'] - 2.0*self.root
        return J


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


class MP_Point(Group):

    def __init__(self, root=1.0):
        super(MP_Point, self).__init__()

        self.add('p', IndepVarComp('x', val=0.0))
        self.add('c', Parab1D(root=root))
        self.connect('p.x', 'c.x')


class TestMPIOpt(MPITestCase, ConcurrentTestCaseMixin):

    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.concurrent_setUp(prefix='test_mpi_opt-')

    def tearDown(self):
        self.concurrent_tearDown()

    def test_parab_FD(self):

        model = Problem(impl=impl)
        root = model.root = Group()
        par = root.add('par', ParallelGroup())

        par.add('c1', Parab1D(root=2.0))
        par.add('c2', Parab1D(root=3.0))

        root.add('p1', IndepVarComp('x', val=0.0))
        root.add('p2', IndepVarComp('x', val=0.0))
        root.connect('p1.x', 'par.c1.x')
        root.connect('p2.x', 'par.c2.x')

        root.add('sumcomp', ExecComp('sum = x1+x2'))
        root.connect('par.c1.y', 'sumcomp.x1')
        root.connect('par.c2.y', 'sumcomp.x2')

        driver = model.driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False
        driver.add_desvar('p1.x', lower=-100, upper=100)
        driver.add_desvar('p2.x', lower=-100, upper=100)
        driver.add_objective('sumcomp.sum')

        root.deriv_options['type'] = 'fd'

        model.setup(check=False)
        model.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, model['p1.x'], 2.0, 1.e-6)
            assert_rel_error(self, model['p2.x'], 3.0, 1.e-6)

    def test_parab_FD_subbed_Pcomps(self):

        model = Problem(impl=impl)
        root = model.root = Group()
        par = root.add('par', ParallelGroup())

        par.add('s1', MP_Point(root=2.0))
        par.add('s2', MP_Point(root=3.0))

        root.add('sumcomp', ExecComp('sum = x1+x2'))
        root.connect('par.s1.c.y', 'sumcomp.x1')
        root.connect('par.s2.c.y', 'sumcomp.x2')

        driver = model.driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False
        driver.add_desvar('par.s1.p.x', lower=-100, upper=100)
        driver.add_desvar('par.s2.p.x', lower=-100, upper=100)
        driver.add_objective('sumcomp.sum')

        root.deriv_options['type'] = 'fd'

        model.setup(check=False)
        model.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, model['par.s1.p.x'], 2.0, 1.e-6)

        if not MPI or self.comm.rank == 1:
            assert_rel_error(self, model['par.s2.p.x'], 3.0, 1.e-6)

    def test_parab_subbed_Pcomps(self):

        model = Problem(impl=impl)
        root = model.root = Group()
        root.ln_solver = lin_solver()

        par = root.add('par', ParallelGroup())

        par.add('s1', MP_Point(root=2.0))
        par.add('s2', MP_Point(root=3.0))

        root.add('sumcomp', ExecComp('sum = x1+x2'))
        root.connect('par.s1.c.y', 'sumcomp.x1')
        root.connect('par.s2.c.y', 'sumcomp.x2')

        driver = model.driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False
        driver.add_desvar('par.s1.p.x', lower=-100, upper=100)
        driver.add_desvar('par.s2.p.x', lower=-100, upper=100)
        driver.add_objective('sumcomp.sum')

        model.setup(check=False)
        model.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, model['par.s1.p.x'], 2.0, 1.e-6)

        if not MPI or self.comm.rank == 1:
            assert_rel_error(self, model['par.s2.p.x'], 3.0, 1.e-6)


class TestMPIOpt2(MPITestCase, ConcurrentTestCaseMixin):

    N_PROCS = 2

    def test_parab_2d(self):
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

    def test_parab_2d_doe(self):
        # create DOE problem for 2D parabola
        doe = Problem(impl=impl)
        root = doe.root = Group()
        driver = doe.driver

        root.add('p', IndepVarComp([('xroot', 0.0), ('yroot', 0.0), ('x', 0.0), ('y', 0.0)]))
        root.add('c', Parab2D())

        driver = CaseDriver(nsamples=2, num_par_doe=2)
        driver.add_desvar('p.xroot', lower=-100, upper=100)
        driver.add_desvar('p.yroot', lower=-100, upper=100)
        driver.add_desvar('p.x', lower=-100, upper=100)
        driver.add_desvar('p.y', lower=-100, upper=100)
        driver.add_response('c.z')

        cases = [
            [('p.xroot', 2.0), ('p.yroot', 3.0), ('p.x', 0.0), ('p.y', 0.0)],
            [('p.xroot', 3.0), ('p.yroot', 5.0), ('p.x', 0.0), ('p.y', 0.0)],
        ]

        # create optimization problem using DOE
        prob = Problem(impl=impl)
        root = prob.root = Group()
        driver = prob.driver

        root.add('p', IndepVarComp([('x', 0.0), ('y', 0.0)]))
        root.add('doe', SubProblem(doe, params=['x', 'y'], unknowns=['z']))

        root.connect('p.x', 'doe.x')
        root.connect('p.y', 'doe.y')

        driver = prob.driver = pyOptSparseDriver()
        driver.options['optimizer'] = OPTIMIZER
        driver.options['print_results'] = False
        driver.add_desvar('p.x',  lower=-100, upper=100)
        driver.add_desvar('p.y1', lower=-100, upper=100)
        driver.add_desvar('p.y2', lower=-100, upper=100)
        driver.add_objective('sumcomp.sum')

        prob.setup(check=False)
        prob.run()

        from pprint import pprint

        num_cases = 0
        for responses, success, msg in prob.driver.get_responses():
            responses = dict(responses)
            pprint(responses)
            num_cases += 1

        # if not MPI or self.comm.rank == 0:
        #     print("sum:", prob['sumcomp.sum'])
        #     print(prob['par.c1.x'], prob['par.c1.y'])

        # from pprint import pprint
        # inputs = []
        # for case in runList:
        #     case = dict(case)
        #     inputs.append((case['p.x'], case['p.y1'], case['p.y2']))
        # pprint(inputs)


class ParallelMPIOptAsym(MPITestCase, ConcurrentTestCaseMixin):
    """The model here has one constraint down inside a Group under a ParallelGroup,
    and one constraint at the top level.
    """
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.concurrent_setUp(prefix='test_mpi_opt_asym-')

        prob = Problem(impl=impl)
        root = prob.root = Group()
        #root.ln_solver = lin_solver()  # this works too (PetscKSP)
        root.ln_solver = LinearGaussSeidel()

        par = root.add('par', ParallelGroup())
        par.ln_solver = LinearGaussSeidel()

        ser1 = par.add('ser1', Group())
        ser1.ln_solver = LinearGaussSeidel()

        ser1.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        ser1.add('comp', SimpleArrayComp(), promotes=['*'])
        ser1.add('con', ExecComp4Test('c = y - 20.0',  # lin_delay=.1,
                                      c=np.array([0.0, 0.0]),
                                      y=np.array([0.0, 0.0])),
                 promotes=['c', 'y'])
        ser1.add('obj', ExecComp4Test('o = y[0]',  # lin_delay=.1,
                                      y=np.array([0.0, 0.0])),
                 promotes=['y', 'o'])

        ser2 = par.add('ser2', Group())
        ser2.ln_solver = LinearGaussSeidel()
        ser2.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['*'])
        ser2.add('comp', SimpleArrayComp(), promotes=['*'])
        ser2.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])),
                 promotes=['y', 'o'])

        root.add('con', ExecComp('c = y - 30.0', c=np.array([0.0, 0.0]),
                                 y=np.array([0.0, 0.0])))
        root.add('total', ExecComp('obj = x1 + x2'))

        root.connect('par.ser1.o', 'total.x1')
        root.connect('par.ser2.o', 'total.x2')
        root.connect('par.ser2.y', 'con.y')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('par.ser1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('par.ser2.x', lower=-50.0, upper=50.0)

        prob.driver.add_objective('total.obj')
        prob.driver.add_constraint('par.ser1.c', equals=0.0)
        prob.driver.add_constraint('con.c', equals=0.0)

        self.prob = prob

    def tearDown(self):
        self.concurrent_tearDown()

    def test_parallel_array_comps_asym_fwd(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser1.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser2.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_array_comps_asym_rev(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.par.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser1.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser2.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)


class ParallelMPIOptPromoted(MPITestCase, ConcurrentTestCaseMixin):
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.concurrent_setUp(prefix='test_mpi_opt_prom-')

        prob = Problem(impl=impl)
        root = prob.root = Group()
        #root.ln_solver = lin_solver()
        root.ln_solver = LinearGaussSeidel()
        par = root.add('par', ParallelGroup())
        par.ln_solver = LinearGaussSeidel()

        ser1 = par.add('ser1', Group())
        ser1.ln_solver = LinearGaussSeidel()

        ser1.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['x'])
        ser1.add('comp', SimpleArrayComp(), promotes=['x', 'y'])
        ser1.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]),
                                 y=np.array([0.0, 0.0])), promotes=['c', 'y'])
        ser1.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])),
                 promotes=['y', 'o'])

        ser2 = par.add('ser2', Group())
        ser2.ln_solver = LinearGaussSeidel()

        ser2.add('p1', IndepVarComp('x', np.zeros([2])), promotes=['x'])
        ser2.add('comp', SimpleArrayComp(), promotes=['x', 'y'])
        ser2.add('con', ExecComp('c = y - 30.0', c=np.array([0.0, 0.0]),
                                 y=np.array([0.0, 0.0])), promotes=['c', 'y'])
        ser2.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])),
                 promotes=['o', 'y'])

        root.add('total', ExecComp('obj = x1 + x2'))

        root.connect('par.ser1.o', 'total.x1')
        root.connect('par.ser2.o', 'total.x2')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('par.ser1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('par.ser2.x', lower=-50.0, upper=50.0)

        prob.driver.add_objective('total.obj')
        prob.driver.add_constraint('par.ser1.c', equals=0.0)
        prob.driver.add_constraint('par.ser2.c', equals=0.0)

        self.prob = prob

    def tearDown(self):
        self.concurrent_tearDown()

    def test_parallel_array_comps_rev(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.par.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser1.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser2.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_derivs_rev(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.par.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser1.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser2.ln_solver.options['mode'] = 'rev'
        prob.driver.parallel_derivs(['par.ser1.c', 'par.ser2.c'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_array_comps_fwd(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser1.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser2.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_derivs_fwd(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser1.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser2.ln_solver.options['mode'] = 'fwd'
        prob.driver.parallel_derivs(['par.ser1.x', 'par.ser2.x'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)


class ParallelMPIOpt(MPITestCase, ConcurrentTestCaseMixin):
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.concurrent_setUp(prefix='par_mpi_opt-')

        prob = Problem(impl=impl)
        root = prob.root = Group()
        #root.ln_solver = lin_solver()
        root.ln_solver = LinearGaussSeidel()
        par = root.add('par', ParallelGroup())
        par.ln_solver = LinearGaussSeidel()

        ser1 = par.add('ser1', Group())
        ser1.ln_solver = LinearGaussSeidel()

        ser1.add('p1', IndepVarComp('x', np.zeros([2])))
        ser1.add('comp', SimpleArrayComp())
        ser1.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]),
                                 y=np.array([0.0, 0.0])))
        ser1.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])))

        ser2 = par.add('ser2', Group())
        ser2.ln_solver = LinearGaussSeidel()

        ser2.add('p1', IndepVarComp('x', np.zeros([2])))
        ser2.add('comp', SimpleArrayComp())
        ser2.add('con', ExecComp('c = y - 30.0', c=np.array([0.0, 0.0]),
                                 y=np.array([0.0, 0.0])))
        ser2.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])))

        root.add('total', ExecComp('obj = x1 + x2'))

        ser1.connect('p1.x', 'comp.x')
        ser1.connect('comp.y', 'con.y')
        ser1.connect('comp.y', 'obj.y')
        root.connect('par.ser1.obj.o', 'total.x1')

        ser2.connect('p1.x', 'comp.x')
        ser2.connect('comp.y', 'con.y')
        ser2.connect('comp.y', 'obj.y')
        root.connect('par.ser2.obj.o', 'total.x2')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('par.ser1.p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('par.ser2.p1.x', lower=-50.0, upper=50.0)

        prob.driver.add_objective('total.obj')
        prob.driver.add_constraint('par.ser1.con.c', equals=0.0)
        prob.driver.add_constraint('par.ser2.con.c', equals=0.0)

        self.prob = prob

    def tearDown(self):
        self.concurrent_tearDown()

    def test_parallel_array_comps_rev(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.par.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser1.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser2.ln_solver.options['mode'] = 'rev'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_derivs_rev(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'rev'
        prob.root.par.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser1.ln_solver.options['mode'] = 'rev'
        prob.root.par.ser2.ln_solver.options['mode'] = 'rev'
        prob.driver.parallel_derivs(['par.ser1.con.c', 'par.ser2.con.c'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_array_comps_fwd(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser1.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser2.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)

    def test_parallel_derivs_fwd(self):
        prob = self.prob
        prob.root.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser1.ln_solver.options['mode'] = 'fwd'
        prob.root.par.ser2.ln_solver.options['mode'] = 'fwd'
        prob.driver.parallel_derivs(['par.ser1.p1.x', 'par.ser2.p1.x'])

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['total.obj'], 50.0, 1e-6)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

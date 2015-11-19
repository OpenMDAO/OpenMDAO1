""" Testing out MPI optimization with pyopt_sparse"""

import os
import unittest
import numpy as np

from openmdao.api import IndepVarComp, ExecComp, LinearGaussSeidel, Component, \
    ParallelGroup, Problem, Group
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error
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
OPT = None
OPTIMIZER = None

try:
    from pyoptsparse import OPT
    try:
        OPT('SNOPT')
        OPTIMIZER = 'SNOPT'
    except:
        try:
            OPT('SLSQP')
            OPTIMIZER = 'SLSQP'
        except:
            pass
except:
    pass

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
        """ Doesn't do much. """
        unknowns['y'] = (params['x'] - self.root)**2 + 7.0

    def linearize(self, params, unknowns, resids):
        """ derivs """
        J = {}
        J['y', 'x'] = 2.0*params['x'] - 2.0*self.root
        return J


class MP_Point(Group):

    def __init__(self, root=1.0):
        super(MP_Point, self).__init__()

        self.add('p', IndepVarComp('x', val=0.0))
        self.add('c', Parab1D(root=root))
        self.connect('p.x', 'c.x')


class TestMPIOpt(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

    def tearDown(self):
        try:
            os.remove('SLSQP.out')
        except OSError:
            pass

        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

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

        root.fd_options['force_fd'] = True

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

        root.fd_options['force_fd'] = True

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


class ParallelMPIOptAsym(MPITestCase):
    """The model here has one constraint down inside a Group under a ParallelGroup,
    and one constraint at the top level.
    """
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

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
        try:
            os.remove('SLSQP.out')
        except OSError:
            pass

        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

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


class ParallelMPIOptPromoted(MPITestCase):
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

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
        try:
            os.remove('SLSQP.out')
        except OSError:
            pass

        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

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


class ParallelMPIOpt(MPITestCase):
    N_PROCS = 2

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

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
        try:
            os.remove('SLSQP.out')
        except OSError:
            pass

        try:
            os.remove('SNOPT_print.out')
            os.remove('SNOPT_summary.out')
        except OSError:
            pass

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

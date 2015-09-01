""" Testing out MPI optimization with pyopt_sparse"""

import unittest
import numpy as np

from openmdao.components import ParamComp, ExecComp
from openmdao.core import Component, ParallelGroup, Problem, Group
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error
from openmdao.test.simple_comps import SimpleArrayComp

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    from openmdao.solvers.petsc_ksp import PetscKSP as lin_solver
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    from openmdao.solvers.scipy_gmres import ScipyGMRES as lin_solver

SKIP = False
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    # Just so python can parse this file.
    from openmdao.core.driver import Driver
    pyOptSparseDriver = Driver
    SKIP = True


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

    def jacobian(self, params, unknowns, resids):
        """ derivs """
        J = {}
        J['y', 'x'] = 2.0*params['x'] - 2.0*self.root
        return J


class MP_Point(Group):

    def __init__(self, root=1.0):
        super(MP_Point, self).__init__()

        self.add('p', ParamComp('x', val=0.0))
        self.add('c', Parab1D(root=root))
        self.connect('p.x', 'c.x')


class TestMPIOpt(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if SKIP:
            raise unittest.SkipTest('Could not import pyOptSparseDriver. Is pyoptsparse installed?')


    def test_parab_FD(self):

        model = Problem(impl=impl)
        root = model.root = Group()
        par = root.add('par', ParallelGroup())

        par.add('c1', Parab1D(root=2.0))
        par.add('c2', Parab1D(root=3.0))

        root.add('p1', ParamComp('x', val=0.0))
        root.add('p2', ParamComp('x', val=0.0))
        root.connect('p1.x', 'par.c1.x')
        root.connect('p2.x', 'par.c2.x')

        root.add('sumcomp', ExecComp('sum = x1+x2'))
        root.connect('par.c1.y', 'sumcomp.x1')
        root.connect('par.c2.y', 'sumcomp.x2')

        driver = model.driver = pyOptSparseDriver()
        driver.add_param('p1.x', low=-100, high=100)
        driver.add_param('p2.x', low=-100, high=100)
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
        driver.add_param('par.s1.p.x', low=-100, high=100)
        driver.add_param('par.s2.p.x', low=-100, high=100)
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
        driver.add_param('par.s1.p.x', low=-100, high=100)
        driver.add_param('par.s2.p.x', low=-100, high=100)
        driver.add_objective('sumcomp.sum')

        model.setup(check=False)
        model.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, model['par.s1.p.x'], 2.0, 1.e-6)

        if not MPI or self.comm.rank == 1:
            assert_rel_error(self, model['par.s2.p.x'], 3.0, 1.e-6)


# class ParallelMPIOpt(MPITestCase):
#     N_PROCS = 2
#
#     def test_parallel_array_comps(self):
#
#         prob = Problem(impl=impl)
#         root = prob.root = Group()
#         par = root.add('par', ParallelGroup())
#
#         par1 = par.add('par1', Group())
#         par1.add('p1', ParamComp('x', np.zeros([2])), promotes=['*'])
#         par1.add('comp', SimpleArrayComp(), promotes=['*'])
#         par1.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
#         par1.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])
#
#         par2 = par.add('par2', Group())
#         par2.add('p1', ParamComp('x', np.zeros([2])), promotes=['*'])
#         par2.add('comp', SimpleArrayComp(), promotes=['*'])
#         par2.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
#         par2.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])
#
#         root.add('total', ExecComp('obj = x1 + x2'))
#
#         root.connect('par.par1.o', 'total.x1')
#         root.connect('par.par2.o', 'total.x2')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.add_param('par.par1.x', low=-50.0, high=50.0)
#         prob.driver.add_param('par.par2.x', low=-50.0, high=50.0)
#
#         prob.driver.add_objective('total.obj')
#         prob.driver.add_constraint('par.par1.c', ctype='eq')
#         prob.driver.add_constraint('par.par2.c', ctype='eq')
#
#         prob.setup(check=False)
#         prob.run()
#
#         assert_rel_error(self, prob['total.obj'], 40.0, 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

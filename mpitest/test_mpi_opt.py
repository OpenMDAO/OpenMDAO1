""" Testing out MPI optimization with pyopt_sparse"""

import unittest
import numpy as np

from openmdao.components import ParamComp, ExecComp
from openmdao.core import Component, ParallelGroup, Problem, Group
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl

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


class TestMPIOpt(MPITestCase):

    N_PROCS = 2

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

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

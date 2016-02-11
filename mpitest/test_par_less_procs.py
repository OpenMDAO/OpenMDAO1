import sys
import numpy
import time

from openmdao.api import Problem, IndepVarComp, Group, ParallelGroup, \
                         Component
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

class Adder(Component):
    def __init__(self, n):
        super(Adder, self).__init__()
        for i in range(1,n+1):
            self.add_param("x%d"%i, 0.0)
        self.add_output("sum", 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns["sum"] = sum(params[n] for n in self._init_params_dict)

class NDiamondPar(Group):
    """ Topology one - n - one."""

    def __init__(self, n, expr="y=2.0*x", req_procs=(1,1)):
        super(NDiamondPar, self).__init__()

        self.add('src', IndepVarComp('x', 2.0))
        self.add('sink', Adder(n))

        sub = self.add('par', ParallelGroup())
        for i in range(1,n+1):
            sub.add("C%d"%i, ExecComp4Test(expr, nl_delay=0.5,
                                           req_procs=req_procs))
            self.connect("src.x", "par.C%d.x"%i)
            self.connect("par.C%d.y"%i, "sink.x%d"%i)

class Test4Par1Proc(MPITestCase):
    """Testing when we have less processors than we need for fully parallel
    execution.
    """

    N_PROCS = 1

    def test_less1(self):
        num_pars = 4

        p = Problem(root=NDiamondPar(num_pars), impl=impl)
        p.setup(check=False)
        start = time.time()
        p.run()

        expected = 4.0*num_pars
        self.assertEqual(p['sink.sum'], expected)

class Test4Par2Proc(MPITestCase):
    """Testing when we have less processors than we need for fully parallel
    execution.
    """

    N_PROCS = 2

    def test_less2(self):
        num_pars = 4

        p = Problem(root=NDiamondPar(num_pars), impl=impl)
        p.setup(check=False)
        start = time.time()
        p.run()

        expected = 4.0*num_pars
        self.assertEqual(p['sink.sum'], expected)

    def test_error(self):
        num_pars = 4

        p = Problem(root=NDiamondPar(num_pars, req_procs=(3,3)), impl=impl)
        try:
            p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                 "This problem was given 2 MPI processes, but it requires "
                 "between 3 and 12.")


class Test4Par3Proc(MPITestCase):
    """Testing when we have less processors than we need for fully parallel
    execution.
    """

    N_PROCS = 3

    def test_less3(self):
        num_pars = 4

        p = Problem(root=NDiamondPar(num_pars), impl=impl)
        p.setup(check=False)
        start = time.time()
        p.run()

        expected = 4.0*num_pars
        self.assertEqual(p['sink.sum'], expected)

class Test4Par4Proc(MPITestCase):
    """Testing when we have less processors than we need for fully parallel
    execution.
    """

    N_PROCS = 4

    def test_4(self):
        num_pars = 4

        p = Problem(root=NDiamondPar(num_pars), impl=impl)
        p.setup(check=False)
        start = time.time()
        p.run()

        expected = 4.0*num_pars
        self.assertEqual(p['sink.sum'], expected)

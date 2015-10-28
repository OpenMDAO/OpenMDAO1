
from __future__ import print_function

import time
import numpy as np
from unittest import TestCase

from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, \
    Component, ParallelFDGroup
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl


class ScalableComp(Component):

    def __init__(self, size, mult=2.0, add=1.0, delay=1.0):
        super(ScalableComp, self).__init__()

        self._size = size
        self._mult = mult
        self._add = add
        self._delay = delay

        self._ncalls = 0

        self.add_param('x', np.zeros(size))
        self.add_output('y', np.zeros(size))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """
        if self._ncalls > 0:  # only delay during FD, not initial run
            time.sleep(self._delay)
        self._ncalls += 1
        unknowns['y'] = params['x']*self._mult + self._add


def setup_1comp_model(par_fds, size, mult, add, delay):
    prob = Problem(impl=impl)
    if par_fds == 1:  # do serial
        prob.root = Group()
    else:
        prob.root = ParallelFDGroup(par_fds)
    prob.root.add('P1', IndepVarComp('x', np.ones(size)))
    prob.root.add('C1', ScalableComp(size, mult, add, delay))

    prob.root.connect('P1.x', 'C1.x')

    prob.driver.add_desvar('P1.x')
    prob.driver.add_objective('C1.y')

    prob.setup(check=False)
    prob.run()

    return prob


def setup_diamond_model(par_fds, size, delay,
                        root_class=ParallelFDGroup,
                        par_class=ParallelGroup):
    prob = Problem(impl=impl)
    if root_class is ParallelFDGroup:
        prob.root = root_class(par_fds)
    else:
        prob.root = root_class()
    root = prob.root

    root.add('P1', IndepVarComp('x', np.ones(size)))

    if par_class is ParallelFDGroup:
        par = root.add("par", par_class(par_fds))
    else:
        par = root.add("par", par_class())

    par.add('C1', ExecComp4Test('y=2.0*x+1.0',
                                nl_delay=delay,
                                x=np.zeros(size), y=np.zeros(size)))
    par.add('C2', ExecComp4Test('y=3.0*x+5.0',
                                nl_delay=delay,
                                x=np.zeros(size), y=np.zeros(size)))

    root.add('C3', ExecComp4Test('y=-3.0*x1+4.0*x2+1.0',
                                 nl_delay=delay,
                                 x1=np.zeros(size), x2=np.zeros(size),
                                 y=np.zeros(size)))

    root.connect("P1.x", "par.C1.x")
    root.connect("P1.x", "par.C2.x")

    root.connect("par.C1.y", "C3.x1")
    root.connect("par.C2.y", "C3.x2")

    prob.driver.add_desvar('P1.x')
    prob.driver.add_objective('C3.y')

    prob.setup(check=False)
    prob.run()

    return prob


class SerialSimpleFDTestCase(TestCase):

    def test_serial_fd(self):
        size = 15
        mult = 2.0
        add = 1.0
        delay = 0.1
        prob = setup_1comp_model(1, size, mult, add, delay)

        J = prob.calc_gradient(['P1.x'], ['C1.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class ParallelSimpleFDTestCase2(MPITestCase):

    N_PROCS = 2

    def test_parallel_fd2(self):
        size = 15
        mult = 2.0
        add = 1.0
        delay = 0.1
        prob = setup_1comp_model(self.N_PROCS, size, mult, add, delay)

        J = prob.calc_gradient(['P1.x'], ['C1.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class ParallelFDTestCase5(MPITestCase):

    N_PROCS = 5

    def test_parallel_fd5(self):
        size = 15
        mult = 2.0
        add = 1.0
        delay = 0.1
        prob = setup_1comp_model(self.N_PROCS, size, mult, add, delay)

        J = prob.calc_gradient(['P1.x'], ['C1.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class SerialDiamondFDTestCase(TestCase):

    def test_diamond_fd(self):
        size = 15
        delay = .1
        prob = setup_diamond_model(1, size, delay)
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.calc_gradient(['P1.x'], ['C3.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_bad_num_par_fds(self):
        try:
            setup_diamond_model(0, 10, 0.1)
        except Exception as err:
            self.assertEquals(str(err), "'': num_par_fds must be >= 1 but value is 0.")


class ParallelDiamondFDTestCase(MPITestCase):

    N_PROCS = 4

    def test_diamond_fd(self):
        size = 15
        delay = .1
        prob = setup_diamond_model(2, size, delay)
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.calc_gradient(['P1.x'], ['C3.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_nested_par_fd(self):
        size = 15
        delay = .1
        prob = setup_diamond_model(4, size, delay, root_class=Group,
                                   par_class=ParallelFDGroup)
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.calc_gradient(['P1.x'], ['C3.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_num_fd_bigger_than_psize(self):
        size = 1
        delay = .1
        prob = setup_diamond_model(2, size, delay)
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.calc_gradient(['P1.x'], ['C3.y'], mode='fd',
                               return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()

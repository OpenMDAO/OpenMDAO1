""" Test out some crucial linear GS tests in parallel."""

from __future__ import print_function

import numpy as np

from openmdao.core.mpiwrap import MPI, MultiProcFailCheck
from openmdao.core.group import Group
from openmdao.core.parallelgroup import ParallelGroup
from openmdao.core.problem import Problem
from openmdao.components.paramcomp import ParamComp
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.test.mpiunittest import MPITestCase
from openmdao.test.simplecomps import FanOutGrouped, FanInGrouped
from openmdao.test.execcomp4test import ExecComp4Test
from openmdao.test.testutil import assert_rel_error
from openmdao.devtools.debug import debug

if MPI:
    from openmdao.core.petscimpl import PetscImpl as impl
else:
    from openmdao.core.basicimpl import BasicImpl as impl


class MatMatTestCase(MPITestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets(self):

        top = Problem(impl=impl)
        top.root = FanInGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # Parallel Groups
        top.driver.add_param('p1.x1')
        top.driver.add_param('p2.x2')
        top.driver.add_objective('comp3.y')

        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets(self):

        top = Problem(impl=impl)
        top.root = FanOutGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # Parallel Groups
        top.driver.add_param('p.x')
        top.driver.add_constraint('c2.y')
        top.driver.add_constraint('c3.y')

        top.setup()
        top.run()

        unknown_list = ['c2.y', 'c3.y']
        param_list = ['p.x']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_in_parallel_sets(self):

        top = Problem(impl=impl)
        top.root = FanInGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # auto calculated mode is fwd, so we don't have to set it explicitly
        # in the ln_solvers in order to have our voi subvecs allocated
        # properly.

        # Parallel Groups
        top.driver.add_param('p1.x1')
        top.driver.add_param('p2.x2')
        top.driver.add_objective('comp3.y')

        # make sure we can't mix inputs and outputs in parallel sets
        try:
            top.driver.group_vars_of_interest(['p1.x1','comp3.y'])
        except Exception as err:
            self.assertEqual(str(err),
               "['p1.x1', 'comp3.y'] cannot be grouped because ['p1.x1'] are "
               "params and ['comp3.y'] are not.")
        else:
            self.fail("Exception expected")

        top.driver.group_vars_of_interest(['p1.x1','p2.x2'])

        self.assertEqual(top.driver.params_of_interest(),
                         [('p1.x1','p2.x2')])

        # make sure we can't add a VOI to multiple groups
        try:
            top.driver.group_vars_of_interest(['p1.x1','p3.x3'])
        except Exception as err:
            self.assertEqual(str(err),
               "'p1.x1' cannot be added to VOI set ('p1.x1', 'p3.x3') "
               "because it already exists in VOI set: ('p1.x1', 'p2.x2')")
        else:
            self.fail("Exception expected")

        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_parallel_sets(self):

        top = Problem(impl=impl)
        top.root = FanOutGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # need to set mode to rev before setup. Otherwise the sub-vectors
        # for the parallel set vars won't get allocated.
        top.root.ln_solver.options['mode'] = 'rev'
        top.root.sub.ln_solver.options['mode'] = 'rev'

        # Parallel Groups
        top.driver.add_param('p.x')
        top.driver.add_constraint('c2.y')
        top.driver.add_constraint('c3.y')
        top.driver.group_vars_of_interest(['c2.y','c3.y'])

        self.assertEqual(top.driver.outputs_of_interest(),
                         [('c2.y','c3.y')])
        top.setup()
        top.run()

        unknown_list = ['c2.y', 'c3.y']
        param_list = ['p.x']

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)


class MatMatIndicesTestCase(MPITestCase):

    N_PROCS = 2

    def test_indices(self):
        asize = 3
        top = Problem(root=Group(), impl=impl)
        root = top.root
        root.ln_solver = LinearGaussSeidel()
        root.ln_solver.options['mode'] = 'rev'

        p = root.add('p', ParamComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add('G1', ParallelGroup())
        G1.ln_solver = LinearGaussSeidel()
        G1.ln_solver.options['mode'] = 'rev'

        c2 = G1.add('c2', ExecComp4Test('y = x * 2.0',
                                   x=np.zeros(asize), y=np.zeros(asize)))
        c3 = G1.add('c3', ExecComp4Test('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
                                   x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add('c4', ExecComp4Test('y = x * 4.0',
                                   x=np.zeros(asize), y=np.zeros(asize)))
        c5 = root.add('c5', ExecComp4Test('y = x * 5.0',
                                   x=np.zeros(asize), y=np.zeros(asize)))

        top.driver.add_param('p.x', indices=[1,2])
        top.driver.add_constraint('c4.y', indices=[1])
        top.driver.add_constraint('c5.y', indices=[2])
        top.driver.group_vars_of_interest(['c4.y','c5.y'])

        root.connect('p.x', 'G1.c2.x')
        root.connect('p.x', 'G1.c3.x')
        root.connect('G1.c2.y','c4.x')
        root.connect('G1.c3.y','c5.x')

        top.setup()
        top.run()

        J = top.calc_gradient(['p.x'],
                              ['c4.y','c5.y'],
                              mode='fwd', return_format='dict')

        assert_rel_error(self, J['c5.y']['p.x'][0], np.array([20.,25.]), 1e-6)
        assert_rel_error(self, J['c4.y']['p.x'][0], np.array([8.,0.]), 1e-6)

        J = top.calc_gradient(['p.x'],
                              ['c4.y','c5.y'],
                              mode='rev', return_format='dict')

        assert_rel_error(self, J['c5.y']['p.x'][0], np.array([20.,25.]), 1e-6)
        assert_rel_error(self, J['c4.y']['p.x'][0], np.array([8.,0.]), 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpiunittest import mpirun_tests
    mpirun_tests()

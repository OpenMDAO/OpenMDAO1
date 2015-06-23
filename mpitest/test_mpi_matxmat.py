""" Test out some crucial linear GS tests in parallel."""

from __future__ import print_function

from openmdao.core.mpiwrap import MPI, MultiProcFailCheck
from openmdao.core.parallelgroup import ParallelGroup
from openmdao.core.problem import Problem
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.test.mpiunittest import MPITestCase
from openmdao.test.simplecomps import FanOutGrouped, FanInGrouped
from openmdao.core.mpiwrap import MPI, MultiProcFailCheck
from openmdao.test.testutil import assert_rel_error

if MPI:
    from openmdao.core.petscimpl import PetscImpl as impl
else:
    from openmdao.core.basicimpl import BasicImpl as impl


class MPITestsMatxMat(MPITestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets(self):

        top = Problem(impl=impl)
        top.root = FanInGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # Parallel Groups
        top.driver._inputs_of_interest = [('p1.x1', ), ('p2.x2', )]
        top.driver._outputs_of_interest = ['comp3.y']

        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets(self):

        top = Problem(impl=impl)
        top.root = FanOutGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # Parallel Groups
        top.driver._outputs_of_interest = [('c2.y', ), ('c3.y', )]
        top.driver._inputs_of_interest = ['p.x']

        top.setup()
        top.run()

        unknown_list = ['c2.y', 'c3.y']
        param_list = ['p.x']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        #assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        #assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

    def test_fan_in_parallel_sets(self):

        top = Problem(impl=impl)
        top.root = FanInGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver = LinearGaussSeidel()

        # Parallel Groups
        top.driver._inputs_of_interest = [('p1.x1', 'p2.x2')]
        top.driver._outputs_of_interest = ['comp3.y']

        top.setup()
        top.run()

        param_list = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_parallel_sets(self):

        #top = Problem(impl=impl)
        #top.root = FanOutGrouped()
        #top.root.ln_solver = LinearGaussSeidel()
        ##top.root.ln_solver.options['mode'] = 'rev'
        #top.root.sub.ln_solver = LinearGaussSeidel()
        ##top.root.sub.ln_solver.options['mode'] = 'rev'

        ## Parallel Groups
        #top.driver._outputs_of_interest = [('c2.y', 'c3.y', )]
        #top.driver._inputs_of_interest = ['p.x']

        #top.setup()
        #top.run()

        #unknown_list = ['c2.y', 'c3.y']
        #param_list = ['p.x']

        #J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        ##assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
        ##assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)
        #assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

        # Have to run a second time, this time adjoint.

        top = Problem(impl=impl)
        top.root = FanOutGrouped()
        top.root.ln_solver = LinearGaussSeidel()
        top.root.ln_solver.options['mode'] = 'rev'
        top.root.sub.ln_solver = LinearGaussSeidel()
        top.root.sub.ln_solver.options['mode'] = 'rev'

        # Parallel Groups
        top.driver._outputs_of_interest = [('c2.y', 'c3.y', )]
        top.driver._inputs_of_interest = ['p.x']

        top.setup()
        top.run()

        unknown_list = ['c2.y', 'c3.y']
        param_list = ['p.x']

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['c2.y']['p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['p.x'][0][0], 15.0, 1e-6)

if __name__ == '__main__':
    from openmdao.test.mpiunittest import mpirun_tests
    mpirun_tests()

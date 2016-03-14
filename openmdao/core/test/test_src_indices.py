""" Testing group-level finite difference. """

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from six import PY3

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp
from openmdao.test.simple_comps import SimpleComp, SimpleArrayComp
from openmdao.test.util import assert_rel_error

if PY3:
    def py3fix(s):
        return s.replace('<type', '<class')
else:
    def py3fix(s):
        return s


class TestSrcIndices(unittest.TestCase):

    def test_src_indices(self):
        size = 10

        root = Group()

        root.add('P1', IndepVarComp('x', np.zeros(size)))
        root.add('C1', ExecComp('y = x * 2.', y=np.zeros(size//2), x=np.zeros(size//2)))
        root.add('C2', ExecComp('y = x * 3.', y=np.zeros(size//2), x=np.zeros(size//2)))

        root.connect('P1.x', "C1.x", src_indices=list(range(size//2)))
        root.connect('P1.x', "C2.x", src_indices=list(range(size//2, size)))

        prob = Problem(root)
        prob.setup(check=False)

        root.P1.unknowns['x'][0:size//2] += 1.0
        root.P1.unknowns['x'][size//2:size] -= 1.0

        prob.run()

        assert_rel_error(self, root.C1.params['x'], np.ones(size//2), 0.0001)
        assert_rel_error(self, root.C2.params['x'], -np.ones(size//2), 0.0001)

    def test_array_to_scalar(self):
        root = Group()

        root.add('P1', IndepVarComp('x', np.array([2., 3.])))
        root.add('C1', SimpleComp())
        root.add('C2', ExecComp('y = x * 3.', y=0., x=0.))

        root.connect('P1.x', 'C1.x', src_indices=[0,])
        root.connect('P1.x', 'C2.x', src_indices=[1,])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(root.C1.params['x'], 2.)
        self.assertAlmostEqual(root.C2.params['x'], 3.)

    def test_subarray_to_promoted_var(self):
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3., 4., 5.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A  = G.add('A', SimpleArrayComp())
        G2 = G.add('G2', Group())

        A2 = G2.add('A2', SimpleArrayComp())

        root.connect('P.x', 'G.A.x', src_indices=[0,1])
        root.connect('P.x', 'C.x', src_indices=[2,])
        root.connect('P.x', 'G.G2.A2.x', src_indices=[3, 4])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, root.G.A.params['x'], np.array([1., 2.]), 0.0001)
        self.assertAlmostEqual(root.C.params['x'], 3.)
        assert_rel_error(self, root.G.G2.A2.params['x'], np.array([4., 5.]), 0.0001)

        # now try the same thing with promoted var
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3., 4., 5.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A  = G.add('A', SimpleArrayComp(), promotes=['x', 'y'])
        G2 = G.add('G2', Group())

        A2 = G2.add('A2', SimpleArrayComp(), promotes=['x', 'y'])

        root.connect('P.x', 'G.x', src_indices=[0,1])
        root.connect('P.x', 'C.x', src_indices=[2,])
        root.connect('P.x', 'G.G2.x', src_indices=[3, 4])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, root.G.A.params['x'], np.array([1., 2.]), 0.0001)
        self.assertAlmostEqual(root.C.params['x'], 3.)
        assert_rel_error(self, root.G.G2.A2.params['x'], np.array([4., 5.]), 0.0001)

    def test_src_indices_connect_error(self):
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3., 4., 5.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A  = G.add('A', SimpleArrayComp())

        root.connect('P.x', 'G.A.x', src_indices=[0])
        root.connect('P.x', 'C.x', src_indices=[2,])

        prob = Problem(root)
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = py3fix("Size 1 of the indexed sub-part of source 'P.x' "
                          "must be the same as size 2 of target 'G.A.x'.")
        self.assertTrue(expected in str(cm.exception))

        # now try the same thing with promoted var
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3., 4., 5.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A  = G.add('A', SimpleArrayComp(), promotes=['x', 'y'])

        root.connect('P.x', 'G.x', src_indices=[0,1,2])
        root.connect('P.x', 'C.x', src_indices=[2,])

        prob = Problem(root)
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = py3fix("Size 3 of the indexed sub-part of source 'P.x' "
                          "must be the same as size 2 of target 'G.A.x' (G.x).")
        self.assertTrue(expected in str(cm.exception))

    def test_inner_connection(self):
        class Squarer(Component):
            def __init__(self, size):
                super(Squarer, self).__init__()
                self.add_param(name='input:x', val=np.zeros(size), desc='x')
                self.add_output(name='output:x2', val=np.zeros(size), desc='x squared')

            def solve_nonlinear(self,params,unknowns,resids):
                unknowns['output:x2'] = params['input:x']**2

        class Cuber(Component):
            def __init__(self, size):
                super(Cuber, self).__init__()
                self.add_param(name='x', val=np.zeros(size), desc='x')
                self.add_output(name='output:x3', val=np.zeros(size), desc='x squared')

            def solve_nonlinear(self,params,unknowns,resids):
                unknowns['output:x3'] = params['x']**3

        class InnerGroup(Group):
            def __init__(self):
                super(InnerGroup, self).__init__()

                self.add('square1', Squarer(5))
                self.add('square2', Squarer(3), promotes=['input:x'])

                # the following connection should result in 'cube1.x' using the
                # same src_indices as 'input:x', which is [2,3,4] from the outer
                # connection
                self.add('cube1', Cuber(3))
                self.connect('input:x', 'cube1.x')

                # the following connection should result in 'cube2.x' using
                # src_indices [0,1] of 'input:x', which corresponds to the
                # src_indices [2,3] from the outer connection
                self.add('cube2', Cuber(2))
                self.connect('input:x', 'cube2.x', src_indices=[0,1])

                # the following connection should result in 'cube3.x' using
                # src_indices [1,2] of 'square1.input:x', which corresponds to the
                # src_indices [1,2] from the outer connection
                self.add('cube3', Cuber(2))
                self.connect('square1.input:x', 'cube3.x', src_indices=[1,2])

        class OuterGroup(Group):
            def __init__(self):
                super(OuterGroup, self).__init__()

                iv = IndepVarComp('input:x', np.zeros(5))
                self.add('indep_vars', iv, promotes=['*'])

                self.add('inner', InnerGroup())
                self.connect('input:x', 'inner.square1.input:x')
                self.connect('input:x', 'inner.input:x', src_indices=[2,3,4])

        prob = Problem(root=OuterGroup())
        prob.setup(check=False)

        prob['input:x'] = np.array([4., 5., 6., 7., 8.])
        prob.run()

        assert_rel_error(self, prob.root.inner.square1.params['input:x'],
                         np.array([4., 5., 6., 7., 8.]), 0.00000001)

        assert_rel_error(self, prob.root.inner.cube1.params['x'],
                         np.array([6., 7., 8.]), 0.00000001)

        assert_rel_error(self, prob.root.inner.cube2.params['x'],
                         np.array([6., 7.]), 0.00000001)

        assert_rel_error(self, prob.root.inner.cube3.params['x'],
                         np.array([5., 6.]), 0.00000001)

    def test_cannonball_src_indices(self):
        # this test replicates the structure of a problem in pointer. The bug was that
        # the state variables in the segments were not getting connected to the proper
        # src_indices of the parameters from the independent variables component
        state_var_names = ['x', 'y', 'vx', 'vy']
        param_arg_names = ['g']
        num_seg = 3
        seg_ncn = 3
        num_nodes = 3

        class Trajectory(Group):
            def __init__(self):
                super(Trajectory, self).__init__()

        class Phase(Group):
            def __init__(self, num_seg, seg_ncn):
                super(Phase, self).__init__()

                ncn_u = 7
                state_vars = [('X_c:{0}'.format(state_name), np.zeros(ncn_u))
                              for state_name in state_var_names]
                self.add('state_var_comp', IndepVarComp(state_vars), promotes=['*'])

                param_args = [('P_s:{0}'.format(param_name), 0.)
                              for param_name in param_arg_names]
                self.add('static_params', IndepVarComp(param_args), promotes=['*'])

                for i in range(num_seg):
                    self.add('seg{0}'.format(i), Segment(seg_ncn))

                offset_states = 0
                for i in range(num_seg):
                    idxs_states = range(offset_states, num_nodes+offset_states)
                    offset_states += num_nodes-1
                    for state_name in state_var_names:
                        self.connect( 'X_c:{0}'.format(state_name), 'seg{0:d}.X_c:{1}'.format(i, state_name), src_indices=idxs_states)
                    for param_name in param_arg_names:
                        self.connect( 'P_s:{0}'.format(param_name), 'seg{0:d}.P_s:{1}'.format(i, param_name))

        class Segment(Group):
            def __init__(self, num_nodes):
                super(Segment, self).__init__()
                self.add('eom_c', EOM(num_nodes))
                self.add('static_bcast', StaticBCast(num_nodes), promotes=['*'])
                self.add('state_interp', StateInterp(num_nodes), promotes=['*'])

                for name in state_var_names:
                    self.connect('X_c:{0}'.format(name), 'eom_c.X:{0}'.format(name))

        class EOM(Component):
            def __init__(self, num_nodes):
                super(EOM, self).__init__()
                for name in state_var_names:
                    self.add_param('X:{0}'.format(name), np.zeros(num_nodes))
                    self.add_output('dXdt:{0}'.format(name), np.zeros(num_nodes))
                for name in param_arg_names:
                    self.add_param('P:{0}'.format(name), 0.)

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['dXdt:x'][:] = params['X:vx']
                unknowns['dXdt:y'][:] = params['X:vy']
                unknowns['dXdt:vx'][:] = 0.0
                unknowns['dXdt:vy'][:] = -params['P:g']

        class StaticBCast(Component):
            def __init__(self, num_nodes):
                super(StaticBCast, self).__init__()
                for name in param_arg_names:
                    self.add_param('P_s:{0}'.format(name), 0.)

            def solve_nonlinear(self, params, unknowns, resids):
                pass

        class StateInterp(Component):
            def __init__(self, num_nodes):
                super(StateInterp, self).__init__()
                for name in state_var_names:
                    self.add_param('X_c:{0}'.format(name), np.zeros(num_nodes))

            def solve_nonlinear(self, params, unknowns, resids):
                pass

        prob = Problem(root=Trajectory())
        phase0 = prob.root.add('phase0', Phase(num_seg, seg_ncn))

        # Call setup so we can access variables through the prob dict interface
        prob.setup(check=False)

        # Populate the unique cardinal values of the states with some values we expect to be distributed to the phases
        prob['phase0.X_c:x'][:] = [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ]
        prob['phase0.X_c:y'][:] = [ 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
        prob['phase0.X_c:vx'][:] = [ 0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0 ]
        prob['phase0.X_c:vy'][:] = [ 0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0 ]
        prob['phase0.P_s:g'] = 9.80665

        # Run to update the values throughout the model
        prob.run()

        for state in ['x','y','vx','vy']:
            phase_cardinal_values = prob['phase0.X_c:{0}'.format(state)]
            idx = 0
            #print('phase0.X_c:{0}'.format(state), phase_cardinal_values)
            for i in range(num_seg):
                seg_ncn = num_nodes
                seg_cardinal_values = prob['phase0.seg{0}.X_c:{1}'.format(i, state)]
                eomc_cardinal_values = prob['phase0.seg{0}.eom_c.X:{1}'.format(i, state)]
                #print('phase0.seg{0}.X_c:{1}'.format(i, state), seg_cardinal_values)
                #print('phase0.seg{0}.eom_c.X:{1}'.format(i, state), eomc_cardinal_values)
                assert_almost_equal(seg_cardinal_values, phase_cardinal_values[idx:idx+seg_ncn], decimal=12)
                assert_almost_equal(seg_cardinal_values, eomc_cardinal_values, decimal=12)
                idx = idx+seg_ncn-1

    def test_slice_conversion(self):
        p = Problem(root=Group())
        size = 11
        c1 = p.root.add("C1", ExecComp(["c=a+b","d=a-b"],
                                            a=np.zeros(size),
                                            b=np.zeros(size),
                                            c=np.zeros(size),
                                            d=np.zeros(size)))

        size = 3
        c2 = p.root.add("C2", ExecComp(["c=a+b","d=a-b"],
                                            a=np.zeros(size),
                                            b=np.zeros(size),
                                            c=np.zeros(size),
                                            d=np.zeros(size)))

        # make two array connections with src_indices having stride > 1
        p.root.connect("C1.c", "C2.a", src_indices=[1,3,5])  # slice (1,7,2)
        p.root.connect("C1.c", "C2.b", src_indices=[6,8,10])  # slice(6,12,2)

        p.setup(check=False)
        p.run()

    def test_duplicate_src_indices(self):
        size = 10

        root = Group()

        root.add('P1', IndepVarComp('x', np.zeros(size//2)))
        root.add('C1', ExecComp('y = x**2', y=np.zeros(size), x=np.zeros(size)))

        root.connect('P1.x', "C1.x", src_indices=2*list(range(size//2)))

        prob = Problem(root)
        prob.setup(check=False)

        prob["P1.x"] = np.arange(5,dtype=float)

        prob.run()

        r = np.arange(5, dtype=float)**2
        expected = np.concatenate((r, r))

        assert_almost_equal( prob["C1.y"], expected, decimal=7)


if __name__ == "__main__":
    unittest.main()

""" Test out the 'scaler' metadata, which allows a user to scale an unknown
or residual on the way in."""
from __future__ import print_function

import unittest
from six import iteritems
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, ScipyGMRES, Newton
from openmdao.test.util import assert_rel_error
from openmdao.test.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives


class BasicComp(Component):
    """ Simple component to demonstrate scaling an unknown."""

    def __init__(self):
        super(BasicComp, self).__init__()

        # Params
        self.add_param('x', 2000.0)

        # Unknowns
        self.add_output('y', 6000.0, scaler=1000.0)

        self.store_y = 0.0

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 3.0*params['x']
        self.store_y = unknowns['y']

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}
        J[('y', 'x')] = np.array([[3.0]])
        return J


class SimpleImplicitComp(Component):
    """ A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol: when x = 0.5, z = 2.666

    Coupled derivs:

    y = x + 8/(x+1)
    dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

    z = 4/(x+1)
    dz_dx = -4/(x+1)**2 = -1.7777777777777777
    """

    def __init__(self, scaler=10.0, resid_scaler=1.0):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0, scaler=scaler, resid_scaler=resid_scaler)

        self.maxiter = 10
        self.atol = 1.0e-12

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method)."""

        x = params['x']
        z = unknowns['z']
        znew = z

        iter = 0
        eps = 1.0e99
        while iter < self.maxiter and abs(eps) > self.atol:
            z = znew
            znew = 4.0 - x*z

            eps = x*znew + znew - 4.0

        unknowns['z'] = znew
        unknowns['y'] = x + 2.0*znew

        resids['z'] = eps
        #print(unknowns['y'], unknowns['z'])

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']
        #print(x, unknowns['y'], z, resids['z'], resids['y'])

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}

        # Output equation
        J[('y', 'x')] = np.array([1.0])
        J[('y', 'z')] = np.array([2.0])

        # State equation
        J[('z', 'z')] = np.array([params['x'] + 1.0])
        J[('z', 'x')] = np.array([unknowns['z']])

        return J

class SimpleImplicitCompApply(SimpleImplicitComp):
    """ Use apply_lineaer instead."""

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            if 'y' in dresids:
                if 'x' in dparams:
                    dresids['y'] += dparams['x']
                if 'z' in dunknowns:
                    dresids['y'] += 2.0*dunknowns['z']

            if 'z' in dresids:
                if 'x' in dparams:
                    dresids['z'] += (np.array([unknowns['z']])).dot(dparams['x'])
                if 'z' in dunknowns:
                    dresids['z'] += (np.array([params['x'] + 1.0])).dot(dunknowns['z'])

        elif mode == 'rev':
            #dparams['x'] = self.multiplier*dresids['y']
            if 'y' in dresids:
                if 'x' in dparams:
                    dparams['x'] += dresids['y']
                if 'z' in dunknowns:
                    dunknowns['z'] += 2.0*dresids['y']

            if 'z' in dresids:
                if 'x' in dparams:
                    dparams['x'] += (np.array([unknowns['z']])).dot(dresids['z'])
                if 'z' in dunknowns:
                    dunknowns['z'] += (np.array([params['x'] + 1.0])).dot(dresids['z'])


class StateConnection(Component):
    """ Define connection with an explicit equation. This version allows
    scaling of state and residual."""

    def __init__(self, resid_scaler=1.0):
        super(StateConnection, self).__init__()

        # Inputs
        self.add_param('y2_actual', 1.0)

        # States
        self.add_state('y2_command', val=1.0, resid_scaler=resid_scaler)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        y2_actual = params['y2_actual']
        y2_command = unknowns['y2_command']

        resids['y2_command'] = y2_actual - y2_command

    def solve_nonlinear(self, params, unknowns, resids):
        """ This is a dummy comp that doesn't modify its state."""
        pass

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}

        # State equation
        J[('y2_command', 'y2_command')] = -1.0
        J[('y2_command', 'y2_actual')] = 1.0

        return J


class SellarStateConnection(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self, resid_scaler=1.0):
        super(SellarStateConnection, self).__init__()

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub = self.add('sub', Group(), promotes=['x', 'z', 'y1', 'state_eq.y2_actual',
                                                 'state_eq.y2_command', 'd1.y2', 'd2.y2'])
        sub.ln_solver = ScipyGMRES()

        subgrp = sub.add('state_eq_group', Group(), promotes=['state_eq.y2_actual',
                                                              'state_eq.y2_command'])
        subgrp.ln_solver = ScipyGMRES()
        subgrp.add('state_eq', StateConnection(resid_scaler=resid_scaler))

        sub.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        self.connect('state_eq.y2_command', 'd1.y2')
        self.connect('d2.y2', 'state_eq.y2_actual')

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['x', 'z', 'y1', 'obj'])
        self.connect('d2.y2', 'obj_cmp.y2')

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        self.connect('d2.y2', 'con_cmp2.y2')

        self.nl_solver = Newton()


class ArrayComp2D(Component):
    """2D Array component."""

    def __init__(self):
        super(ArrayComp2D, self).__init__()

        # Params
        self.add_param('x', np.zeros((2, 2)))

        # Unknowns
        self.add_output('y', np.zeros((2, 2)), scaler=5.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much."""

        x = params['x']
        y = unknowns['y']

        y[0][0] = 2.0*x[0][0] + 1.0*x[0][1] + \
                  3.0*x[1][0] + 7.0*x[1][1]

        y[0][1] = 4.0*x[0][0] + 2.0*x[0][1] + \
                  6.0*x[1][0] + 5.0*x[1][1]

        y[1][0] = 3.0*x[0][0] + 6.0*x[0][1] + \
                  9.0*x[1][0] + 8.0*x[1][1]

        y[1][1] = 1.0*x[0][0] + 3.0*x[0][1] + \
                  2.0*x[1][0] + 4.0*x[1][1]

        unknowns['y'] = y

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}
        J['y', 'x'] = np.array([[2.0, 1.0, 3.0, 7.0],
                                [4.0, 2.0, 6.0, 5.0],
                                [3.0, 6.0, 9.0, 8.0],
                                [1.0, 3.0, 2.0, 4.0]])
        return J


class TestVecWrapperScaler(unittest.TestCase):

    def test_basic(self):
        top = Problem()
        root = top.root = Group()
        root.add('p', IndepVarComp('x', 2000.0))
        root.add('comp1', BasicComp())
        root.add('comp2', ExecComp(['y = 2.0*x']))
        root.connect('p.x', 'comp1.x')
        root.connect('comp1.y', 'comp2.x')

        top.driver.add_desvar('p.x', 2000.0)
        top.driver.add_objective('comp2.y')

        root.comp1.fd_options['extra_check_partials_form'] = 'complex_step'

        top.setup(check=False)
        top.run()

        # correct execution, scale does not affect downstream output
        assert_rel_error(self, top['comp2.y'], 12000.0, 1e-6)

        # in-component query is unscaled
        assert_rel_error(self, root.comp1.store_y, 6000.0, 1e-6)

        # afterwards direct query is unscaled
        assert_rel_error(self, root.unknowns['comp1.y'], 6000.0, 1e-6)

        # OpenMDAO behind-the-scenes query is scaled
        # (So, internal storage is scaled)
        assert_rel_error(self, root.unknowns._dat['comp1.y'].val, 6.0, 1e-6)

        # Correct derivatives
        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fwd')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='rev')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fd')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        # Clean up old FD
        top.run()

        # Make sure check_partials works too
        data = top.check_partial_derivatives(out_stream=None)
        #data = top.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        # Clean up old FD
        top.run()

        # Make sure check_totals works too
        data = top.check_total_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            assert_rel_error(self, val1['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][2], 0.0, 1e-5)

    def test_basic_gmres(self):
        top = Problem()
        root = top.root = Group()
        root.add('p', IndepVarComp('x', 2000.0))
        root.add('comp1', BasicComp())
        root.add('comp2', ExecComp(['y = 2.0*x']))
        root.connect('p.x', 'comp1.x')
        root.connect('comp1.y', 'comp2.x')

        top.driver.add_desvar('p.x', 2000.0)
        top.driver.add_objective('comp2.y')

        root.ln_solver = ScipyGMRES()

        top.setup(check=False)
        top.run()

        # correct execution
        assert_rel_error(self, top['comp2.y'], 12000.0, 1e-6)

        # in-component query is unscaled
        assert_rel_error(self, root.comp1.store_y, 6000.0, 1e-6)

        # afterwards direct query is unscaled
        assert_rel_error(self, root.unknowns['comp1.y'], 6000.0, 1e-6)

        # OpenMDAO behind-the-scenes query is scaled
        # (So, internal storage is scaled)
        assert_rel_error(self, root.unknowns._dat['comp1.y'].val, 6.0, 1e-6)

        # Correct derivatives
        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fwd')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='rev')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fd')
        assert_rel_error(self, J[0][0], 6.0, 1e-6)

        # Clean up old FD
        top.run()

        # Make sure check_partials works too
        data = top.check_partial_derivatives(out_stream=None)
        #data = top.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.root.comp.fd_options['extra_check_partials_form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        # Correct total derivatives (we can do this one manually)
        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        # Clean up old FD
        prob.run()

        # Partials
        data = prob.check_partial_derivatives(out_stream=None)
        #data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 20.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 15.0, 1e-6)

        # Clean up old FD
        prob.run()

        # Make sure check_totals works too
        data = prob.check_total_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            assert_rel_error(self, val1['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit_resid(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp(resid_scaler=0.001))
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()
        #print(prob.root.comp.resids['z'])

        # Correct total derivatives (we can do this one manually)
        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        # Clean up old FD
        prob.run()

        # Partials
        data = prob.check_partial_derivatives(out_stream=None)
        #data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-3)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 20.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667/0.001, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 15.0/0.001, 1e-6)

    def test_simple_implicit_apply(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitCompApply())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        # Correct total derivatives (we can do this one manually)
        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-6)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='rev')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-6)

        J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fd')
        assert_rel_error(self, J[0][0], -2.5555511, 1e-6)

        # Clean up old FD
        prob.run()

        # Partials
        data = prob.check_partial_derivatives(out_stream=None)
        #data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 20.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 15.0, 1e-6)

    def test_apply_linear_units(self):
        # Make sure we can index into dparams

        class Attitude_Angular(Component):
            """ Calculates angular velocity vector from the satellite's orientation
            matrix and its derivative.
            """

            def __init__(self, n=2):
                super(Attitude_Angular, self).__init__()

                self.n = n

                # Inputs
                self.add_param('O_BI', np.zeros((3, 3, n)), units="ft",
                               desc="Rotation matrix from body-fixed frame to Earth-centered "
                               "inertial frame over time")

                self.add_param('Odot_BI', np.zeros((3, 3, n)), units="km",
                               desc="First derivative of O_BI over time")

                # Outputs
                self.add_output('w_B', np.zeros((3, n)), units="1/s", scaler=12.0,
                                desc="Angular velocity vector in body-fixed frame over time")

                self.dw_dOdot = np.zeros((n, 3, 3, 3))
                self.dw_dO = np.zeros((n, 3, 3, 3))

            def solve_nonlinear(self, params, unknowns, resids):
                """ Calculate output. """

                O_BI = params['O_BI']
                Odot_BI = params['Odot_BI']
                w_B = unknowns['w_B']

                for i in range(0, self.n):
                    w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
                    w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
                    w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

                #print(unknowns['w_B'])

            def linearize(self, params, unknowns, resids):
                """ Calculate and save derivatives. (i.e., Jacobian) """

                O_BI = params['O_BI']
                Odot_BI = params['Odot_BI']

                for i in range(0, self.n):
                    self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
                    self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

                    self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
                    self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

                    self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
                    self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

            def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
                """ Matrix-vector product with the Jacobian. """

                dw_B = dresids['w_B']

                if mode == 'fwd':
                    for k in range(3):
                        for i in range(3):
                            for j in range(3):
                                if 'O_BI' in dparams:
                                    dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                        dparams['O_BI'][i, j, :]
                                if 'Odot_BI' in dparams:
                                    dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                        dparams['Odot_BI'][i, j, :]

                else:

                    for k in range(3):
                        for i in range(3):
                            for j in range(3):

                                if 'O_BI' in dparams:
                                    dparams['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                        dw_B[k, :]

                                if 'Odot_BI' in dparams:
                                    dparams['Odot_BI'][i, j, :] -= -self.dw_dOdot[:, k, i, j] * \
                                        dw_B[k, :]

        prob = Problem()
        root = prob.root = Group()
        prob.root.add('comp', Attitude_Angular(n=5), promotes=['*'])
        prob.root.add('p1', IndepVarComp('O_BI', np.ones((3, 3, 5))), promotes=['*'])
        prob.root.add('p2', IndepVarComp('Odot_BI', np.ones((3, 3, 5))), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        indep_list = ['O_BI', 'Odot_BI']
        unknown_list = ['w_B']
        Jf = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                                return_format='dict')

        indep_list = ['O_BI', 'Odot_BI']
        unknown_list = ['w_B']
        Jr = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                                return_format='dict')

        for key, val in iteritems(Jr):
            for key2 in val:
                diff = abs(Jf[key][key2] - Jr[key][key2])
                assert_rel_error(self, diff, 0.0, 1e-10)

        # Clean up old FD
        prob.run()

        # Partials
        data = prob.check_partial_derivatives(out_stream=None)
        #data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_sellar_state_connection(self):

        prob = Problem()
        prob.root = SellarStateConnection()
        prob.root.nl_solver = Newton()

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.root.nl_solver.iter_count, 8)

    def test_array2D(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        #prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = prob.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x']/5.0)
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x']/5.0)
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_converge_diverge(self):

        from openmdao.test.converge_diverge import Comp1, Comp2, Comp3, Comp5, Comp6, Comp7

        class Comp4(Component):

            def __init__(self):
                super(Comp4, self).__init__()
                self.add_param('x1', 1.0)
                self.add_param('x2', 1.0)
                self.add_output('y1', 1.0, scaler = 3.0)
                self.add_output('y2', 1.0, scaler = 2.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """ Runs the component."""
                unknowns['y1'] = params['x1'] + 2.0*params['x2']
                unknowns['y2'] = 3.0*params['x1'] - 5.0*params['x2']

            def linearize(self, params, unknowns, resids):
                """Returns the Jacobian."""
                J = {}
                J[('y1', 'x1')] = 1.0
                J[('y1', 'x2')] = 2.0
                J[('y2', 'x1')] = 3.0
                J[('y2', 'x2')] = -5.0
                return J


        class ConvergeDiverge(Group):
            """ Topology one - two - one - two - one. This model was critical in
            testing parallel reverse scatters."""

            def __init__(self):
                super(ConvergeDiverge, self).__init__()

                self.add('p', IndepVarComp('x', 2.0))

                self.add('comp1', Comp1())
                self.add('comp2', Comp2())
                self.add('comp3', Comp3())
                self.add('comp4', Comp4())
                self.add('comp5', Comp5())
                self.add('comp6', Comp6())
                self.add('comp7', Comp7())

                self.connect("p.x", "comp1.x1")
                self.connect('comp1.y1', 'comp2.x1')
                self.connect('comp1.y2', 'comp3.x1')
                self.connect('comp2.y1', 'comp4.x1')
                self.connect('comp3.y1', 'comp4.x2')
                self.connect('comp4.y1', 'comp5.x1')
                self.connect('comp4.y2', 'comp6.x1')
                self.connect('comp5.y1', 'comp7.x1')
                self.connect('comp6.y1', 'comp7.x2')


        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.setup(check=False)

        prob.run()

        # Partials
        data = prob.check_partial_derivatives(out_stream=None)
        #data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        # Clean up old FD
        prob.run()

        # Make sure check_totals works too
        data = prob.check_total_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            assert_rel_error(self, val1['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][2], 0.0, 1e-5)

    def test_scaler_on_src_with_unit_conversion(self):

        from openmdao.core.test.test_units import TgtCompC, TgtCompF, TgtCompK

        class SrcComp(Component):

            def __init__(self):
                super(SrcComp, self).__init__()

                self.add_param('x1', 100.0)
                self.add_output('x2', 100.0, units='degC', scaler=10.0)

            def solve_nonlinear(self, params, unknowns, resids):
                """ No action."""
                unknowns['x2'] = params['x1']

            def linearize(self, params, unknowns, resids):
                """ Derivative is 1.0"""
                J = {}
                J[('x2', 'x1')] = np.array([1.0])
                return J

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Make sure the scale factor is included here, even though no unit conversion
        self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         (10.0, 0.0))

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Need to clean up after FD gradient call, so just rerun.
        prob.run()

        # Make sure check partials handles conversion
        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

        stream = cStringIO()
        conv = prob.root.list_unit_conv(stream=stream)
        self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)
        self.assertTrue((('src.x2', 'tgtK.x2'), ('degC', 'degK')) in conv)

    def test_errors(self):

        class Comp1(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp1, self).__init__()

                self.add_output('y', 6000.0, resid_scaler=1.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp1()

        msg = ("resid_scaler is only supported for states.")
        self.assertEqual(str(cm.exception), msg)

        class Comp2(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp2, self).__init__()

                self.add_param('y', 6000.0, resid_scaler=1.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp2()

        msg = ("resid_scaler is only supported for states.")
        self.assertEqual(str(cm.exception), msg)

        class Comp3(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp3, self).__init__()

                self.add_param('y', 6000.0, scaler=1.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp3()

        msg = ("scaler is only supported for outputs and states.")
        self.assertEqual(str(cm.exception), msg)

        class Comp4(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp4, self).__init__()

                self.add_output('y', 6000.0, scaler=0.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp4()

        msg = ("scaler value must be nonzero.")
        self.assertEqual(str(cm.exception), msg)

        class Comp4(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp4, self).__init__()

                self.add_output('y', 6000.0, scaler=0.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp4()

        msg = ("scaler value must be nonzero.")
        self.assertEqual(str(cm.exception), msg)

        class Comp5(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp5, self).__init__()

                self.add_state('y', 6000.0, scaler=0.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp5()

        msg = ("scaler value must be nonzero.")
        self.assertEqual(str(cm.exception), msg)

        class Comp6(Component):
            """ Comp with a scaler or resid_scaler that raises an exception."""

            def __init__(self):
                super(Comp6, self).__init__()

                self.add_state('y', 6000.0, resid_scaler=0.0)

        with self.assertRaises(ValueError) as cm:
            z = Comp6()

        msg = ("resid_scaler value must be nonzero.")
        self.assertEqual(str(cm.exception), msg)

if __name__ == "__main__":
    unittest.main()

""" Test out the 'scaler' metadata, which allows a user to scale an unknown
or residual on the way in."""

import unittest
from six import iteritems

import numpy as np

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, ScipyGMRES
from openmdao.test.util import assert_rel_error


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

    def __init__(self):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0, scaler=10.0, resid_scaler=1.0)

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

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']

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
        top.setup(check=False)
        top.run()

        # correct execution
        assert_rel_error(self, top['comp2.y'], 12.0, 1e-6)

        # in-component query is unscaled
        assert_rel_error(self, root.comp1.store_y, 6000.0, 1e-6)

        # local query is unscaled
        assert_rel_error(self, root.unknowns['comp1.y'], 6000.0, 1e-6)

        # OpenMDAO behind-the-scenes query is scaled
        assert_rel_error(self, root.unknowns._dat['comp1.y'].val, 6.0, 1e-6)

        # Correct derivatives
        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fwd')
        print(J)
        #assert_rel_error(self, J[0][0], 0.006, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='rev')
        print(J)
        #assert_rel_error(self, J[0][0], 0.006, 1e-6)

        J = top.calc_gradient(['p.x'], ['comp2.y'], mode='fd')
        print(J)
        #assert_rel_error(self, J[0][0], 0.006, 1e-6)

    def test_simple_implicit(self):

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        data = prob.check_partial_derivatives(out_stream=None)
        data = prob.check_partial_derivatives()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()

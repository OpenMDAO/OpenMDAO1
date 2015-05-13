""" Some simple test components. """

import numpy as np

from openmdao.core.component import Component


class SimpleComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self):
        super(SimpleComp, self).__init__()

        # Params
        self.add_param('x', 3.0)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 2.0*params['x']


class SimpleCompDerivMatVec(SimpleComp):
    """ The simplest component you can imagine, this time with derivatives
    defined using apply_linear. """

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            dresids['y'] = 2.0*dparams['x']

        elif mode == 'rev':
            dparams['x'] = 2.0*dresids['y']


class SimpleCompDerivJac(SimpleComp):
    """ The simplest component you can imagine, this time with derivatives
    defined using Jacobian to return a jacobian. """

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""

        J = {}
        J[('y', 'x')] = np.array(2.0)
        return J


class SimpleArrayComp(Component):
    '''A fairly simple array component'''

    def __init__(self):
        super(SimpleArrayComp, self).__init__()

        # Params
        self.add_param('x', np.zeros([2]))

        # Unknowns
        self.add_output('y', np.zeros([2]))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        params['y'][0] = 2.0*params['x'][0] + 7.0*params['x'][1]
        params['y'][1] = 5.0*params['x'][0] - 3.0*params['x'][1]
        #print "ran", params['x'], params['y']

    def jacobian(self, params, unknowns):
        """Analytical derivatives"""

        dy1_dx1 = 2.0
        dy1_dx2 = 7.0
        dy2_dx1 = 5.0
        dy2_dx2 = -3.0
        J = {}
        J[('y', 'x')] = np.array([[dy1_dx1, dy1_dx2], [dy2_dx1, dy2_dx2]])

        return J


class SimpleImplicitComp(Component):
    """ A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol: when x = 0.5, z = 2.666
    """

    def __init__(self):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5, low=0.01, high=1.0)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0)

        self.maxiter = 10
        self.atol = 1.0e-6

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method) """

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
        """ Don't solve; just calculate the redisual. """

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']

    def jacobian(self, params, unknowns):
        """Analytical derivatives"""

        J = {}

        # Output equation
        J[('y', 'x')] = np.array([1.0])
        J[('y', 'z')] = np.array([2.0])

        # State equation
        J[('z', 'z')] = np.array([params['x'] + 1.0])
        J[('z', 'x')] = np.array([unknowns['z']])

        return J


class SimpleNoflatComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self):
        super(SimpleNoflatComp, self).__init__()

        # Params
        self.add_param('x', '')

        # Unknowns
        self.add_output('y', '')

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = params['x']+self.name


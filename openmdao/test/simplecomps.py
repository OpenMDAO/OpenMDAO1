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

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns,
                 dresids, mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            dunknowns['y'] = 2.0*dparams['x']

        elif mode == 'rev':
            dunknowns['x'] = 2.0*dparams['y']


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
        #print "ran", self.x, self.y

    def jacobian(self, params, unknowns):
        """Analytical first derivatives"""

        dy1_dx1 = 2.0
        dy1_dx2 = 7.0
        dy2_dx1 = 5.0
        dy2_dx2 = -3.0
        J = {}
        J[('y', 'x')] = np.array([[dy1_dx1, dy1_dx2], [dy2_dx1, dy2_dx2]])

        return J

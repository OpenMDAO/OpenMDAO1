""" Some simple test components. """

from openmdao.core.component import Component


class SimpleComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self):
        super(SimpleComp, self).__init__()

        # Params
        self.add_param('x', 3.0)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, outputs, resids):
        """ Doesn't do much. """

        outputs['y'] = 2.0*params['x']


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



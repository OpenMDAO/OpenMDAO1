from openmdao.core.component import Component
import numpy as np

class LinearSystem(Component):

    def __init__(self, size):
        super(LinearSystem, self).__init__()
        self.size = size

        self.add_param("A", val=np.eye(size))
        self.add_param("b", val=np.ones(size))

        self.add_state("x", val=np.zeros(size))

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['x'] = np.linalg.solve(params['A'], params['b'])

    def apply_nonlinear(self, params, unknowns, resids):
        """Evaluating residual for given state."""

        resids['x'] = params['A'].dot(unknowns['x']) - params['b']

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):

        if mode == 'fwd':

            if 'x' in dunknowns:
                dresids['x'] += params['A'].dot(dunknowns['x'])
            if 'A' in dparams:
                dresids['x'] += dparams['A'].dot(unknowns['x'])
            if 'b' in dparams:
                dresids['x'] -= dparams['b']

        elif mode == 'rev':

            if 'x' in dunknowns:
                dunknowns['x'] += params['A'].dot(dresids['x'])
            if 'A' in dparams:
                dparams['A'] += dresids['x']*(unknowns['x'])
            if 'b' in dparams:
                dparams['b'] -= dresids['x']

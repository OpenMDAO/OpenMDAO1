import unittest
import numpy as np
from openmdao.core.component import Component
# from openmdao.components.linear_system import LinearSystem

class LinearSystem(Component):

    def __init__(self):
        pass

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['x'] = np.linalg.solve(params['A'], params['b'])

    def apply_nonlinear(self, params, unknowns, resids):
        ''' Evaluating residual for given state '''

        resids['x'] = params['A'].dot(unknowns['x']) - params['b']

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns, dresids):

        dresids['x'] += params['A'].dot(dunknowns['x'])
        dresids['x'] += dparams['A'].dot(unknowns['x'])
        dresids['x'] -= dparams['b']



class TestLinearSystem(unittest.TestCase):

    def setUp(self):
        self.s = LinearSystem()
        self.params, self.unknowns, self.resids = {}, {}, {}

        self.params['A'] = np.eye(10)
        self.params['b'] = np.ones(10)

        self.unknowns['x'] = np.zeros(10)

        self.resids['x'] = np.zeros(10)

    def test_solve_linearsystem(self):

        self.s.solve_nonlinear(self.params, self.unknowns, self.resids)

        actual = np.ones(10)

        rel = np.linalg.norm(actual - self.unknowns['x']) / np.linalg.norm(actual)

        assert rel < 1e-6

    def test_apply_nonlinear(self):

        self.unknowns['x'] = np.array([0,1,2,3,4,5,6,7,8,9])
        self.s.apply_nonlinear(self.params, self.unknowns, self.resids)

        actual = np.array([0,1,2,3,4,5,6,7,8,9]) - 1

        rel = np.linalg.norm(actual - self.resids['x'])/np.linalg.norm(actual)

        assert rel < 1e-6

    def test_apply_linear(self):

        self.params['A'] = np.eye(2)
        self.params['b'] = np.ones(2)

        self.unknowns['x'] = np.zeros(2)

        self.resids['x'] = np.zeros(2)

        d_params = {}
        d_params['A'] = np.random.random((2, 2))
        d_params['b'] = np.random.random((2))
        d_unknowns = {}
        d_unknowns['x'] = np.random.random((2))
        d_resids = {}
        d_resids['x'] = np.zeros((2))

        self.s.apply_linear(self.params, self.unknowns, self.resids,
                            d_params, d_unknowns, d_resids)

        actual = d_resids['x']

        x = self.unknowns['x']
        piece = np.array([[x[0], x[1], 0, 0], [0, 0, x[0], x[1]]])
        temp = self.params['A'].dot(d_unknowns['x']) + \
               piece.dot(d_params['A'].flatten()) + \
               -d_params['b']

        self.assertLessEqual(np.linalg.norm((actual - temp))/np.linalg.norm(actual), 1e-6)

if __name__ == "__main__":
    unittest.main()

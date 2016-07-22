""" Tests a system with a solve_linear method defined plus preconditioner on gmres. """

import unittest

import numpy as np
from scipy import linalg

from openmdao.api import IndepVarComp, Group, Problem, Component, ScipyGMRES, Newton, LinearGaussSeidel
from openmdao.test.util import assert_rel_error


class SellarInABox(Component):

    def __init__(self):
        super(SellarInABox, self).__init__()

        # Global Design Variable
        self.add_param('z', val=np.zeros(2))

        # Local Design Variable
        self.add_param('x', val=0.)

        # Coupling parameter
        self.add_output('y1', val=1.0)

        # Solver hook
        self.add_state('y2', val=1.0)

        # Calculated value
        self.y2 = 1.0

        self.count_solve_linear = 0

    def solve_nonlinear(self, params, unknowns, resids):
        """ Just calculate unknowns.
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = unknowns['y2']

        unknowns['y1'] = z1**2 + z2 + x1 - 0.2*y2

    def apply_nonlinear(self, params, unknowns, resids):
        """Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        y2 = y1**(.5) + z1 + z2"""

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y1 = unknowns['y1']
        y2 = unknowns['y2']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        resids['y1'] = z1**2 + z2 + x1 - 0.2*y2 - y1
        self.y2 = y1**.5 + z1 + z2
        resids['y2'] = self.y2 - y2

    def linearize(self, params, unknowns, resids):
        """ Jacobian for Sellar discipline 1."""

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y1 = unknowns['y1']
        y2 = unknowns['y2']

        J = {}

        J['y1', 'y2'] = -0.2
        J['y1', 'z'] = np.array([[2.0*z1, 1.0]])
        J['y1', 'x'] = 1.0
        J['y2', 'y1'] = .5*y1**-.5
        J['y2', 'z'] = np.array([[1.0, 1.0]])
        J['y2', 'y2'] = -1.0

        dRdy = np.zeros((2, 2))
        dRdy[0, 1] = J['y1', 'y2']
        dRdy[0, 0] = 1.0
        dRdy[1, 0] = J['y2', 'y1']
        dRdy[1, 1] = J['y2', 'y2']

        # lu factorization for use with solve_linear
        self.lup = linalg.lu_factor(dRdy)

        return J

    def solve_linear(self, dumat, drmat, vois, mode=None):

        self.count_solve_linear += 1

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t=0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t=1

        for voi in vois:
            rhs = np.zeros((2, 1))
            rhs[0] = rhs_vec[voi]['y1']
            rhs[1] = rhs_vec[voi]['y2']

            sol = linalg.lu_solve(self.lup, rhs, trans=t)

            sol_vec[voi]['y1'] = sol[0]
            sol_vec[voi]['y2'] = sol[1]


class TestNLGaussSeidel(unittest.TestCase):

    def test_nested(self):

        top = Problem()
        root = top.root = Group()
        sub = root.add('sub', Group(), promotes=['x', 'z', 'y1', 'y2'])

        sub.add('comp', SellarInABox(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        sub.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        root.nl_solver = Newton()
        root.ln_solver = ScipyGMRES()
        root.ln_solver.preconditioner = LinearGaussSeidel()

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        self.assertGreater(top.root.sub.comp.count_solve_linear, 0)

    def test_flat(self):

        top = Problem()
        root = top.root = Group()

        root.add('comp', SellarInABox(), promotes=['x', 'z', 'y1', 'y2'])
        root.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        root.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        root.nl_solver = Newton()
        root.ln_solver.options['maxiter'] = 5

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['y1'], 25.58830273, .00001)
        assert_rel_error(self, top['y2'], 12.05848819, .00001)

        self.assertGreater(top.root.comp.count_solve_linear, 0)


if __name__ == "__main__":
    unittest.main()

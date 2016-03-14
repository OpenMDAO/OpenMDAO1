""" Test to guide converting over to a correctly-signed residual."""

import unittest

import numpy as np
from scipy.optimize import fsolve

from openmdao.api import Problem, Group, Component, IndepVarComp, ScipyGMRES, ExecComp


class SimpleImplicit(Component):

    def __init__(self):
        super(SimpleImplicit, self).__init__()

        self.add_param('a', shape=1., step_size=1e-3)
        self.add_param('b', shape=1)

        self.add_state('x', val=np.ones(2))

    def apply_nonlinear(self, p, u, r):

        x0,x1 = u['x']
        r['x'][0] = x0**2 - x1**2 + p['a']
        r['x'][1] = p['b']*x0*x1

    def _r(self, x, p, u, r):
        u['x'] = x
        self.apply_nonlinear(p, u, r)
        return r['x']

    def solve_nonlinear(self, p, u, r):
        self.apply_nonlinear(p, u, r)

    def linearize(self, p, u, r):
        J = {}
        x0,x1 = u['x']

        J['x','a'] = np.array([[1.],
                               [0.]])
        J['x','b'] = np.array([[0.],
                               [x0*x1]])
        J['x','x'] = np.array([[2.*x0, -2.*x1],
                               [p['b']*x1, p['b']*x0]])
        self.J = J

        return J

    def solve_linear(self, dumat, drmat, vois, mode=None):
        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        for voi in vois:
            if mode == "fwd":
                sol_vec[voi].vec[:] = np.linalg.solve(self.J['x','x'], -rhs_vec[voi].vec)
            else:
                sol_vec[voi].vec[:] = np.linalg.solve(self.J['x','x'].T, -rhs_vec[voi].vec)


class TestResidual(unittest.TestCase):

    def test_implicit_sign(self):

        p = Problem()
        p.root = Group()

        dvars = ( ('a', 3.), ('b', 10.))
        p.root.add('desvars', IndepVarComp(dvars), promotes=['a', 'b'])

        sg = p.root.add('sg', Group(), promotes=["*"])
        sg.add('si', SimpleImplicit(), promotes=['a', 'b', 'x'])

        p.root.add('func', ExecComp('f = 2*x0+a'), promotes=['f', 'x0', 'a'])
        p.root.connect('x', 'x0', src_indices=[1])

        p.driver.add_objective('f')
        p.driver.add_desvar('a')

        p.setup()
        p0 = np.array([1.5, 2.])
        p['x'] = p0
        p.root.apply_nonlinear(p.root.params, p.root.unknowns, p.root.resids)
        r0 = p.root.resids['x'].copy()

        p1 = np.array([1.5001, 2.00])
        p['x'] = p1
        p.root.apply_nonlinear(p.root.params, p.root.unknowns, p.root.resids)
        r1 = p.root.resids['x'].copy()

        print r0
        print r1
        print "deriv", (r1-r0)/(p1[0]-p0[0])

        p.root.linearize(p.root.params, p.root.unknowns, p.root.resids)

        p.root.dumat[None]['x'][:] = np.array([1.,0.])
        print p.root.dumat[None]['x']
        p.root.clear_dparams()
        p.root._sys_apply_linear('fwd', do_apply=p.root._do_apply, vois=(None, ))
        print p.root.drmat[None]['x']


if __name__ == "__main__":
    unittest.main()

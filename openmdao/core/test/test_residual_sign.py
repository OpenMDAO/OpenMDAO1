""" Test to guide converting over to a correctly-signed residual derivatives."""

import unittest

import numpy as np
from scipy.optimize import fsolve

from openmdao.api import Problem, Group, Component, IndepVarComp, ScipyGMRES, ExecComp
from openmdao.test.util import assert_rel_error


class SimpleImplicitSL(Component):

    def __init__(self):
        super(SimpleImplicitSL, self).__init__()

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


class SimpleExplicit(Component):

    def __init__(self):
        super(SimpleExplicit, self).__init__()

        self.add_param('x', val=np.ones(2))
        self.add_output('resid', val=np.ones(2))

    def solve_nonlinear(self, p, u, r):

        x0,x1 = p['x']
        u['resid'][0] = x0**2 - x1**2
        u['resid'][1] = 10.0*x0*x1

    def linearize(self, p, u, r):
        J = {}
        x0, x1 = p['x']

        J['resid', 'x'] = np.array([[2.*x0, -2.*x1],
                                    [10.0*x1, 10.0*x0]])
        self.J = J
        return J


class TestResidual(unittest.TestCase):

    def test_implicit_sign(self):

        p = Problem()
        p.root = Group()

        dvars = ( ('a', 3.), ('b', 10.))
        p.root.add('desvars', IndepVarComp(dvars), promotes=['a', 'b'])

        sg = p.root.add('sg', Group(), promotes=["*"])
        sg.add('si', SimpleImplicitSL(), promotes=['a', 'b', 'x'])

        p.root.add('func', ExecComp('f = 2*x0+a'), promotes=['f', 'x0', 'a'])
        p.root.connect('x', 'x0', src_indices=[1])

        p.driver.add_objective('f')
        p.driver.add_desvar('a')

        p.setup(check=False)
        p['x'] = np.array([1.5, 2.])

        p.run_once()
        p.root.linearize(p.root.params, p.root.unknowns, p.root.resids)

        # fwd
        p.root.dumat[None]['x'][:] = np.array([1., 0.])
        p.root.clear_dparams()

        p.root._sys_apply_linear('fwd', do_apply=p.root._do_apply, vois=(None, ))
        assert_rel_error(self, p.root.drmat[None]['x'][0], 3.0, 1e-8)
        assert_rel_error(self, p.root.drmat[None]['x'][1], 20.0, 1e-8)

        # rev 1
        p.root.drmat[None].vec[:] = 0.0
        p.root.dumat[None].vec[:] = 0.0
        p.root.clear_dparams()
        p.root.drmat[None]['x'][:] = np.array([1., 0.])

        p.root._sys_apply_linear('rev', do_apply=p.root._do_apply, vois=(None, ))
        assert_rel_error(self, p.root.dumat[None]['x'][0], 3.0, 1e-8)
        assert_rel_error(self, p.root.dumat[None]['a'], 1.0, 1e-8)
        assert_rel_error(self, p.root.dumat[None]['b'], 0.0, 1e-8)

        # rev 2
        p.root.drmat[None].vec[:] = 0.0
        p.root.dumat[None].vec[:] = 0.0
        p.root.clear_dparams()
        p.root.drmat[None]['x'][:] = np.array([0., 1.])

        p.root._sys_apply_linear('rev', do_apply=p.root._do_apply, vois=(None, ))
        assert_rel_error(self, p.root.dumat[None]['x'][0], 20.0, 1e-8)

    def test_explicit_sign(self):

        p = Problem()
        p.root = Group()

        p.root.add('desvars', IndepVarComp('x', np.array([1.5, 2.])), promotes=['x'])

        sg = p.root.add('sg', Group(), promotes=["*"])
        sg.add('si', SimpleExplicit(), promotes=['x', 'resid'])

        p.root.add('func', ExecComp('f = 2*x0 + resid', f=np.zeros((2, )), resid=np.zeros((2, ))),
                   promotes=['f', 'x0', 'resid'])
        p.root.connect('x', 'x0', src_indices=[1])

        p.driver.add_objective('f')
        p.driver.add_desvar('x')

        p.setup(check=False)
        p['x'] = np.array([1.5, 2.])

        p.run_once()
        p.root.linearize(p.root.params, p.root.unknowns, p.root.resids)

        # fwd
        p.root.dumat[None]['x'][:] = np.array([1., 0.])
        p.root.clear_dparams()

        p.root._sys_apply_linear('fwd', do_apply=p.root._do_apply, vois=(None, ))
        assert_rel_error(self, p.root.drmat[None]['resid'][0], 3.0, 1e-8)
        assert_rel_error(self, p.root.drmat[None]['resid'][1], 20.0, 1e-8)

        # rev 1
        p.root.clear_dparams()
        p.root.drmat[None].vec[:] = 0.0
        p.root.dumat[None].vec[:] = 0.0
        p.root.drmat[None]['resid'][:] = np.array([1., 0.])

        p.root._sys_apply_linear('rev', do_apply=p.root._do_apply, vois=(None, ))
        assert_rel_error(self, p.root.dumat[None]['x'][0], 3.0, 1e-8)

        # rev 2
        p.root.clear_dparams()
        p.root.drmat[None].vec[:] = 0.0
        p.root.dumat[None].vec[:] = 0.0
        p.root.drmat[None]['resid'][:] = np.array([0., 1.])

        p.root._sys_apply_linear('rev', do_apply=p.root._do_apply, vois=(None, ))

        assert_rel_error(self, p.root.dumat[None]['x'][0], 20.0, 1e-8)

if __name__ == "__main__":
    unittest.main()

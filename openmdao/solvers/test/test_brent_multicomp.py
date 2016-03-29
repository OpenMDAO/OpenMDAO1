""" Unit test for the Brent, one variable nonlinear solver. In this case, the
system has multiple components."""

import unittest

from openmdao.api import Group, Problem, Component, Brent, ScipyGMRES
from openmdao.test.util import assert_rel_error


class CompPart1(Component):

    def __init__(self):
        super(CompPart1, self).__init__()
        self.fd_options['force_fd'] = True

        self.add_param('a', val=1.)
        self.add_param('n', val=77.0/27.0)
        self.add_param('part2', val=0.)

        self.add_state('x', val=2., lower=0, upper=100)

    def solve_nonlinear(self, p, u, r):
        pass

    def apply_nonlinear(self, p, u, r):
        r['x'] = p['a'] * u['x']**p['n'] + p['part2'] #+ p['b'] * u['x'] - p['c']
        # print self.pathname, "ap_nl", p['part2'], p['a'], u['x'], p['n'], r['x']


class CompPart2(Component):

    def __init__(self):
        super(CompPart2, self).__init__()
        self.fd_options['force_fd'] = True

        self.add_param('b', val=1.)
        self.add_param('c', val=10.)
        self.add_param('x', val=2.)
        self.add_output('part2', val=0.)

    def solve_nonlinear(self, p, u, r):

        u['part2'] = p['b'] * p['x'] - p['c']
        # print self.pathname, "sp_nl", p['x'], p['c'], p['b'], u['part2']


class Combined(Group):

    def __init__(self):
        super(Combined, self).__init__()

        self.add('p1', CompPart1(), promotes=['*'])
        self.add('p2', CompPart2(), promotes=['*'])
        # self.add('i1', IndepVarComp('a', 1.), promotes=['*'])
        # self.add('i2', IndepVarComp('b', 1.), promotes=['*'])

        # self.nl_solver = Newton()
        self.nl_solver = Brent()
        # self.nl_solver.options['iprint'] = 1
        self.nl_solver.options['state_var'] = 'x'

        self.ln_solver = ScipyGMRES()

        self.set_order(('p1','p2'))


class BrentMultiCompTestCase(unittest.TestCase):
    """test to make sure brent can converge multiple components
    in a group with a single residual across them all
    """

    def test_multi_comp(self):
        p = Problem()
        p.root = Combined()
        p.setup(check=False)
        p.run()

        assert_rel_error(self, p.root.unknowns['x'], 2.06720359226, .0001)
        assert_rel_error(self, p.root.resids['x'], 0, .0001)

if __name__ == "__main__":

   unittest.main()
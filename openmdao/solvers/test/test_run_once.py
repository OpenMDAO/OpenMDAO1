""" Unit test for the RunOnce simple nonlinear solver. """

import unittest

from openmdao.api import IndepVarComp, Group, Problem, RunOnce, ExecComp
from openmdao.test.util import assert_rel_error


class TestNLGaussSeidel(unittest.TestCase):

    def test_sellar(self):

        prob = Problem()
        prob.root = Group()
        prob.root.nl_solver = RunOnce()
        prob.root.add('comp', ExecComp('y=x*2.0'))
        prob.root.add('P1', IndepVarComp('x', 3.0))
        prob.root.connect('P1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        y = prob.root.unknowns['comp.y']

        assert_rel_error(self, y, 6.0, .000001)

        # Make sure we aren't iterating like crazy
        self.assertEqual(prob.root.nl_solver.iter_count, 1)


if __name__ == "__main__":
    unittest.main()

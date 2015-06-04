""" Unit test for the RunOnce simple nonlinear solver. """

import unittest

from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.run_once import RunOnce
from openmdao.test.testutil import assert_rel_error

class TestNLGaussSeidel(unittest.TestCase):

    def test_sellar(self):

        top = Problem()
        top.root = Group()
        top.root.nl_solver = RunOnce()
        top.root.add('comp', ExecComp('y=x*2.0'))
        top.root.add('P1', ParamComp('x', 3.0))
        top.root.connect('P1:x', 'comp:x')

        top.setup()
        top.run()

        y = top.root.unknowns['comp:y']

        assert_rel_error(self, y, 6.0, .000001)

        # Make sure we aren't iterating like crazy
        self.assertEqual(top.root.nl_solver.iter_count, 1)

if __name__ == "__main__":
    unittest.main()

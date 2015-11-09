""" Test the examples directory to keep them in working order.

   NOTE: If you make any changes to this file, you must make the corresponding
   change to the example file.
"""

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer
from openmdao.test.util import assert_rel_error

from paraboloid_example import Paraboloid
from paraboloid_optimize_constrained import Paraboloid as ParaboloidOptCon
from paraboloid_optimize_unconstrained import Paraboloid as ParaboloidOptUnCon


class TestExamples(unittest.TestCase):

    def test_paraboloid_example(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.setup(check=False)
        top.run()

        assert_rel_error(self, root.p.unknowns['f_xy'], -15.0, 1e-6)

    def test_paraboloid_optimize_constrained(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', ParaboloidOptCon())

        # Constraint Equation
        root.add('con', ExecComp('c = x-y'))

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')
        root.connect('p.x', 'con.x')
        root.connect('p.y', 'con.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x', low=-50, high=50)
        top.driver.add_desvar('p2.y', low=-50, high=50)
        top.driver.add_objective('p.f_xy')
        top.driver.add_constraint('con.c', lower=15.0)

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['p.x'], 7.166667, 1e-6)
        assert_rel_error(self, top['p.y'], -7.833333, 1e-6)

    def test_paraboloid_optimize_unconstrained(self):

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', ParaboloidOptUnCon())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x', low=-50, high=50)
        top.driver.add_desvar('p2.y', low=-50, high=50)
        top.driver.add_objective('p.f_xy')

        top.setup(check=False)
        top.run()

        assert_rel_error(self, top['p.x'], 6.666667, 1e-6)
        assert_rel_error(self, top['p.y'], -7.333333, 1e-6)

if __name__ == "__main__":
    unittest.main()

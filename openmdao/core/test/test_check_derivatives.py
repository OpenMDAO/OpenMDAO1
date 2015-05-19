""" Testing for Problem.check_partial_derivatives."""

import unittest

from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()

        top.setup()
        top.run()

        data = top.check_partial_derivatives()

        print data


if __name__ == "__main__":
    unittest.main()

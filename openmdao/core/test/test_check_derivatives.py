""" Testing for Problem.check_partial_derivatives."""

from six import iteritems
import unittest

from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDivergeGroups
from openmdao.test.testutil import assert_rel_error


class TestProblemCheckPartials(unittest.TestCase):

    def test_double_diamond_model(self):

        top = Problem()
        top.root = ConvergeDivergeGroups()

        top.setup()
        top.run()

        data = top.check_partial_derivatives()
        #print data

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()

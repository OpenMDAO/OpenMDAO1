""" Unit subtest for Groups.  This one tests OpenMDAO's behavior when connecting
variables of different types."""

import unittest
import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.simplecomps import SimpleComp


class TestGroupConnect(unittest.TestCase):
    """ Unit subtest for Groups.  This one tests OpenMDAO's behavior when connecting
    variables of different types."""

    def setUp(self):
        self.top = Problem(root=Group())
        self.src = self.top.root.add('src', SimpleComp())
        self.tgt = self.top.root.add('tgt', SimpleComp())

    def test_connection_array_array_same_size(self):

        # Array to Array same size

        self.src.add_output('y', np.zeros((2, 1)))
        self.tgt.add_output('x', np.zeros((2, 1)))

        self.top.root.connect('src:y', "tgt:x")

        self.top.setup()
        self.top.run()
        var = self.top.root.varmanager.unknowns['tgt:y']
        self.assertEqual(len(var), 2)
        self.assertEqual(var[0][0], 0.0)
        self.assertEqual(var[1][0], 0.0)


if __name__ == "__main__":
    unittest.main()

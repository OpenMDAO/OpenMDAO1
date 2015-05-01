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
        self.top.root.add('src', SimpleComp())
        self.top.root.add('tgt', SimpleComp())

    def test_connection_vartypes(self):

        # Array to Array same size
        self.top.root.src.add_output('y1', np.zeros((3, 1)))

if __name__ == "__main__":
    unittest.main()

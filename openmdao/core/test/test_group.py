""" Unit test for Groups. """

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.test.testcomps import SimpleComp

class TestGroup(unittest.TestCase):

    def test_add(self):

        group = Group()
        comp = SimpleComp()
        group.add('mycomp', comp)

        subs = dict(group.subsystems())
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs['mycomp'], comp)

        comp2 = SimpleComp()
        group.add("nextcomp", comp2)

        subs = dict(group.subsystems())
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs['mycomp'], comp)
        self.assertEqual(subs['nextcomp'], comp2)

    def test_variables(self):
        group = Group()
        group.add('C1', SimpleComp(), promotes=['x'])
        group.add("C2", SimpleComp(), promotes=['y'])

        params, unknowns, states = group.variables()

        self.assertEqual(params.keys(), ['x', 'C2:x'])
        self.assertEqual(unknowns.keys(), ['C1:y', 'y'])
        self.assertEqual(states.keys(), [])

    def test_connect(self):
        G1 = Group()
        G1.add('C1', SimpleComp()])
        G1.add("C2", SimpleComp(), promotes=['y'])
        G1.connect('C1:y', 'C2:x')

        G2 = Group()
        G2.add(G1, promotes=['y'])
        G2.add("C3", SimpleComp(), promotes=['x'])
        G2.connect('G1:y', 'x')

        G2.setup_vectors()

    def test_setup(self):
        pass

    def test_solve(self):
        pass


if __name__ == "__main__":
    unittest.main()

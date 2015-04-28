""" Unit test for Groups. """

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
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
        G4 = Group()

        G2 = G4.add('G2', Group())
        G2.add('C1', ParamComp('y1', 5.))

        G1 = G2.add('G1', Group())
        G1.add('C2', SimpleComp())

        G3 = G4.add('G3', Group())
        G3.add('C3', SimpleComp())
        G3.add('C4', SimpleComp())

        G2.connect('C1:y1', 'G1:C2:x')
        G4.connect('G2:G1:C2:y', 'G3:C3:x')
        # G4.connect('G2:C1:y', 'G2:G1:C2:x')
        G3.connect('C3:y', 'C4:x')

        self.assertEqual(set(G1.connections()), set([]))
        self.assertEqual(set(G2.connections()), set([('G1:C2:x', 'C1:y1')]))
        self.assertEqual(set(G3.connections()), set([('C4:x', 'C3:y')]))
        self.assertEqual(set(G4.connections()),
            set([('G3:C4:x', 'G3:C3:y'), ('G3:C3:x', 'G2:G1:C2:y'), ('G2:G1:C2:x', 'G2:C1:y1')]))

        G4.setup_vectors()
        # print G4.varmanager.variables()

    def test_setup(self):
        pass

    def test_solve(self):
        pass


if __name__ == "__main__":
    unittest.main()

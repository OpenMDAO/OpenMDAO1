""" Unit test for Groups. """

import unittest

from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp

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
        # G4.connect('G2:C1:y1', 'G2:G1:C2:x')
        G4.connect('G2:G1:C2:y', 'G3:C3:x')
        G3.connect('C3:y', 'C4:x')

        self.assertEqual(set(G1.connections()), set([]))
        self.assertEqual(set(G2.connections()), set([('G1:C2:x', 'C1:y1')]))
        self.assertEqual(set(G3.connections()), set([('C4:x', 'C3:y')]))
        self.assertEqual(set(G4.connections()),
            set([('G3:C4:x', 'G3:C3:y'), ('G3:C3:x', 'G2:G1:C2:y'), ('G2:G1:C2:x', 'G2:C1:y1')]))

        G4.setup_syspaths('')
        G4.setup_vectors()

        expected_G4_params   = ['G3:C3:x']
        expected_G4_unknowns = ['G2:C1:y1', 'G2:G1:C2:y', 'G3:C3:y', 'G3:C4:y']

        expected_G3_params   = ['C4:x']
        expected_G3_unknowns = ['C3:y', 'C4:y']

        expected_G2_params   = ['G1:C2:x']
        expected_G2_unknowns = ['C1:y1', 'G1:C2:y']

        expected_G1_params   = []
        expected_G1_unknowns = ['C2:y']

        self.assertEqual(G4.varmanager.params.keys(),    expected_G4_params)
        self.assertEqual(G4.varmanager.dparams.keys(),   expected_G4_params)
        self.assertEqual(G4.varmanager.unknowns.keys(),  expected_G4_unknowns)
        self.assertEqual(G4.varmanager.dunknowns.keys(), expected_G4_unknowns)
        self.assertEqual(G4.varmanager.resids.keys(),    expected_G4_unknowns)
        self.assertEqual(G4.varmanager.dresids.keys(),   expected_G4_unknowns)

        self.assertEqual(G3.varmanager.params.keys(),    expected_G3_params)
        self.assertEqual(G3.varmanager.dparams.keys(),   expected_G3_params)
        self.assertEqual(G3.varmanager.unknowns.keys(),  expected_G3_unknowns)
        self.assertEqual(G3.varmanager.dunknowns.keys(), expected_G3_unknowns)
        self.assertEqual(G3.varmanager.resids.keys(),    expected_G3_unknowns)
        self.assertEqual(G3.varmanager.dresids.keys(),   expected_G3_unknowns)

        self.assertEqual(G2.varmanager.params.keys(),    expected_G2_params)
        self.assertEqual(G2.varmanager.dparams.keys(),   expected_G2_params)
        self.assertEqual(G2.varmanager.unknowns.keys(),  expected_G2_unknowns)
        self.assertEqual(G2.varmanager.dunknowns.keys(), expected_G2_unknowns)
        self.assertEqual(G2.varmanager.resids.keys(),    expected_G2_unknowns)
        self.assertEqual(G2.varmanager.dresids.keys(),   expected_G2_unknowns)

        self.assertEqual(G1.varmanager.params.keys(),    expected_G1_params)
        self.assertEqual(G1.varmanager.dparams.keys(),   expected_G1_params)
        self.assertEqual(G1.varmanager.unknowns.keys(),  expected_G1_unknowns)
        self.assertEqual(G1.varmanager.dunknowns.keys(), expected_G1_unknowns)
        self.assertEqual(G1.varmanager.resids.keys(),    expected_G1_unknowns)
        self.assertEqual(G1.varmanager.dresids.keys(),   expected_G1_unknowns)

        # verify subsystem is using shared view of parent unknowns vector
        G4.varmanager.unknowns['G2:C1:y1'] = 99.
        self.assertEqual(G2.varmanager.unknowns['C1:y1'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(G4.varmanager.unknowns.metadata('G2:C1:y1'),
                         G2.varmanager.unknowns.metadata('C1:y1'))

    def test_promotes(self):
        # TODO: test groups with components that promote variables
        self.fail("Test not yet implemented")

    def test_setup(self):
        pass

    def test_solve(self):
        pass


if __name__ == "__main__":
    unittest.main()

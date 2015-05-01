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

        params, unknowns = group.setup_variables()

        self.assertEqual(params.keys(), ['x', 'C2:x'])
        self.assertEqual(unknowns.keys(), ['C1:y', 'y'])

    def test_connect(self):
        root = Group()

        G2 = root.add('G2', Group())
        G2.add('C1', ParamComp('y1', 5.))

        G1 = G2.add('G1', Group())
        G1.add('C2', SimpleComp())

        G3 = root.add('G3', Group())
        G3.add('C3', SimpleComp())
        G3.add('C4', SimpleComp())

        G2.connect('C1:y1', 'G1:C2:x')
        #root.connect('G2:C1:y1', 'G2:G1:C2:x')
        root.connect('G2:G1:C2:y', 'G3:C3:x')
        G3.connect('C3:y', 'C4:x')

        root.setup_paths('')

        self.assertEqual(root.pathname, '')
        self.assertEqual(G3.pathname, 'G3')
        self.assertEqual(G2.pathname, 'G2')
        self.assertEqual(G1.pathname, 'G2:G1')

        # verify variables are set up correctly
        root.setup_variables()

        # TODO: check for expected results from setup_variables
        self.assertEqual(G1._params.items(),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'C2:x'})])
        self.assertEqual(G1._unknowns.items(),
                         [('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'C2:y'})])

        self.assertEqual(G2._params.items(),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'G1:C2:x'})])
        self.assertEqual(G2._unknowns.items(),
                         [('G2:C1:y1', {'val': 5.0, 'relative_name': 'C1:y1'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G1:C2:y'})])

        self.assertEqual(G3._params.items(),
                         [('G3:C3:x', {'val': 3.0, 'relative_name': 'C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'C4:x'})])
        self.assertEqual(G3._unknowns.items(),
                         [('G3:C3:y', {'val': 5.5, 'relative_name': 'C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'C4:y'})])

        self.assertEqual(root._params.items(),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'G2:G1:C2:x'}),
                          ('G3:C3:x', {'val': 3.0, 'relative_name': 'G3:C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'G3:C4:x'})])

        self.assertEqual(root._unknowns.items(),
                         [('G2:C1:y1', {'val': 5.0, 'relative_name': 'G2:C1:y1'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G2:G1:C2:y'}),
                          ('G3:C3:y', {'val': 5.5, 'relative_name': 'G3:C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'G3:C4:y'})])

        # verify we get correct connection information
        connections = root.get_connections()
        expected_connections = {
            'G2:G1:C2:x': 'G2:C1:y1',
            'G3:C3:x':    'G2:G1:C2:y',
            'G3:C4:x':    'G3:C3:y'
        }
        self.assertEqual(connections, expected_connections)

        from openmdao.core.problem import assign_parameters
        param_owners = assign_parameters(connections)
        expected_owners = {
            'G3': ['G3:C4:x'],
            '':   ['G3:C3:x'],
            'G2': ['G2:G1:C2:x']
        }
        self.assertEqual(param_owners, expected_owners)

        # verify vectors are set up correctly
        root.setup_vectors(param_owners, connections)

        expected_root_params   = ['G3:C3:x']
        expected_root_unknowns = ['G2:C1:y1', 'G2:G1:C2:y', 'G3:C3:y', 'G3:C4:y']

        expected_G3_params   = ['C4:x']
        expected_G3_unknowns = ['C3:y', 'C4:y']

        expected_G2_params   = ['G1:C2:x']
        expected_G2_unknowns = ['C1:y1', 'G1:C2:y']

        expected_G1_params   = []
        expected_G1_unknowns = ['C2:y']

        self.assertEqual(root.varmanager.params.keys(),    expected_root_params)
        self.assertEqual(root.varmanager.dparams.keys(),   expected_root_params)
        self.assertEqual(root.varmanager.unknowns.keys(),  expected_root_unknowns)
        self.assertEqual(root.varmanager.dunknowns.keys(), expected_root_unknowns)
        self.assertEqual(root.varmanager.resids.keys(),    expected_root_unknowns)
        self.assertEqual(root.varmanager.dresids.keys(),   expected_root_unknowns)

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
        root.varmanager.unknowns['G2:C1:y1'] = 99.
        self.assertEqual(G2.varmanager.unknowns['C1:y1'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(root.varmanager.unknowns.metadata('G2:C1:y1'),
                         G2.varmanager.unknowns.metadata('C1:y1'))


    def test_promotes(self):
        # TODO: test groups with components that promote variables
        self.fail("Test not yet implemented")

    def test_setup(self):
        self.fail("Test not yet implemented")

    def test_solve(self):
        self.fail("Test not yet implemented")


if __name__ == "__main__":
    unittest.main()

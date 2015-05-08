""" Unit test for Groups. """

import unittest

from openmdao.core.group import Group, _get_implicit_connections
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

        # paths must be initialized prior to calling _setup_variables
        group._setup_paths('')
        params_dict, unknowns_dict = group._setup_variables()

        self.assertEqual(list(params_dict.keys()), ['C1:x', 'C2:x'])
        self.assertEqual(list(unknowns_dict.keys()), ['C1:y', 'C2:y'])

        self.assertEqual([m['relative_name'] for n,m in params_dict.items()], ['x', 'C2:x'])
        self.assertEqual([m['relative_name'] for n,m in unknowns_dict.items()], ['C1:y', 'y'])

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

        root._setup_paths('')

        self.assertEqual(root.pathname, '')
        self.assertEqual(G3.pathname, 'G3')
        self.assertEqual(G2.pathname, 'G2')
        self.assertEqual(G1.pathname, 'G2:G1')

        # verify variables are set up correctly
        root._setup_variables()

        # TODO: check for expected results from _setup_variables
        self.assertEqual(list(G1._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'C2:x'})])
        self.assertEqual(list(G1._unknowns_dict.items()),
                         [('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'C2:y'})])

        self.assertEqual(list(G2._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'G1:C2:x'})])
        self.assertEqual(list(G2._unknowns_dict.items()),
                         [('G2:C1:y1', {'val': 5.0, 'relative_name': 'C1:y1'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G1:C2:y'})])

        self.assertEqual(list(G3._params_dict.items()),
                         [('G3:C3:x', {'val': 3.0, 'relative_name': 'C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'C4:x'})])
        self.assertEqual(list(G3._unknowns_dict.items()),
                         [('G3:C3:y', {'val': 5.5, 'relative_name': 'C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'C4:y'})])

        self.assertEqual(list(root._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'G2:G1:C2:x'}),
                          ('G3:C3:x', {'val': 3.0, 'relative_name': 'G3:C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'G3:C4:x'})])

        self.assertEqual(list(root._unknowns_dict.items()),
                         [('G2:C1:y1', {'val': 5.0, 'relative_name': 'G2:C1:y1'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G2:G1:C2:y'}),
                          ('G3:C3:y', {'val': 5.5, 'relative_name': 'G3:C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'G3:C4:y'})])

        # verify we get correct connection information
        connections = root._get_explicit_connections()
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
        root._setup_vectors(param_owners, connections)

        expected_root_params   = ['G3:C3:x']
        expected_root_unknowns = ['G2:C1:y1', 'G2:G1:C2:y', 'G3:C3:y', 'G3:C4:y']

        expected_G3_params   = ['C4:x', 'C3:x']
        expected_G3_unknowns = ['C3:y', 'C4:y']

        expected_G2_params   = ['G1:C2:x']
        expected_G2_unknowns = ['C1:y1', 'G1:C2:y']

        expected_G1_params   = ['C2:x']
        expected_G1_unknowns = ['C2:y']

        self.assertEqual(list(root._varmanager.params.keys()),    expected_root_params)
        self.assertEqual(list(root._varmanager.dparams.keys()),   expected_root_params)
        self.assertEqual(list(root._varmanager.unknowns.keys()),  expected_root_unknowns)
        self.assertEqual(list(root._varmanager.dunknowns.keys()), expected_root_unknowns)
        self.assertEqual(list(root._varmanager.resids.keys()),    expected_root_unknowns)
        self.assertEqual(list(root._varmanager.dresids.keys()),   expected_root_unknowns)

        self.assertEqual(list(G3._varmanager.params.keys()),    expected_G3_params)
        self.assertEqual(list(G3._varmanager.dparams.keys()),   expected_G3_params)
        self.assertEqual(list(G3._varmanager.unknowns.keys()),  expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.dunknowns.keys()), expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.resids.keys()),    expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.dresids.keys()),   expected_G3_unknowns)

        self.assertEqual(list(G2._varmanager.params.keys()),    expected_G2_params)
        self.assertEqual(list(G2._varmanager.dparams.keys()),   expected_G2_params)
        self.assertEqual(list(G2._varmanager.unknowns.keys()),  expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.dunknowns.keys()), expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.resids.keys()),    expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.dresids.keys()),   expected_G2_unknowns)

        self.assertEqual(list(G1._varmanager.params.keys()),    expected_G1_params)
        self.assertEqual(list(G1._varmanager.dparams.keys()),   expected_G1_params)
        self.assertEqual(list(G1._varmanager.unknowns.keys()),  expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.dunknowns.keys()), expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.resids.keys()),    expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.dresids.keys()),   expected_G1_unknowns)

        # verify subsystem is using shared view of parent unknowns vector
        root._varmanager.unknowns['G2:C1:y1'] = 99.
        self.assertEqual(G2._varmanager.unknowns['C1:y1'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(root._varmanager.unknowns.metadata('G2:C1:y1'),
                         G2._varmanager.unknowns.metadata('C1:y1'))

    def test_promotes(self):
        root = Group()

        G2 = root.add('G2', Group())
        G2.add('C1', ParamComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', SimpleComp(), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', SimpleComp())
        G3.add('C4', SimpleComp(), promotes=['x'])

        root.connect('G2:G1:C2:y', 'G3:C3:x')
        G3.connect('C3:y', 'x')

        root._setup_paths('')

        self.assertEqual(root.pathname, '')
        self.assertEqual(G3.pathname, 'G3')
        self.assertEqual(G2.pathname, 'G2')
        self.assertEqual(G1.pathname, 'G2:G1')

        # verify variables are set up correctly
        root._setup_variables()

        # TODO: check for expected results from _setup_variables
        self.assertEqual(list(G1._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'x'})])
        self.assertEqual(list(G1._unknowns_dict.items()),
                         [('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'C2:y'})])

        self.assertEqual(list(G2._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'x'})])
        self.assertEqual(list(G2._unknowns_dict.items()),
                         [('G2:C1:x', {'val': 5.0, 'relative_name': 'x'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G1:C2:y'})])

        self.assertEqual(list(G3._params_dict.items()),
                         [('G3:C3:x', {'val': 3.0, 'relative_name': 'C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'x'})])
        self.assertEqual(list(G3._unknowns_dict.items()),
                         [('G3:C3:y', {'val': 5.5, 'relative_name': 'C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'C4:y'})])

        self.assertEqual(list(root._params_dict.items()),
                         [('G2:G1:C2:x', {'val': 3.0, 'relative_name': 'G2:x'}),
                          ('G3:C3:x', {'val': 3.0, 'relative_name': 'G3:C3:x'}),
                          ('G3:C4:x', {'val': 3.0, 'relative_name': 'x'})])

        self.assertEqual(list(root._unknowns_dict.items()),
                         [('G2:C1:x', {'val': 5.0, 'relative_name': 'G2:x'}),
                          ('G2:G1:C2:y', {'val': 5.5, 'relative_name': 'G2:G1:C2:y'}),
                          ('G3:C3:y', {'val': 5.5, 'relative_name': 'G3:C3:y'}),
                          ('G3:C4:y', {'val': 5.5, 'relative_name': 'G3:C4:y'})])

        # verify we get correct connection information
        connections = root._get_explicit_connections()
        expected_connections = {
            'G3:C3:x':    'G2:G1:C2:y',
            'G3:C4:x':    'G3:C3:y'
        }
        self.assertEqual(connections, expected_connections)

        connections.update(_get_implicit_connections(root._params_dict, root._unknowns_dict))

        from openmdao.core.problem import assign_parameters
        param_owners = assign_parameters(connections)
        expected_owners = {
            'G3': ['G3:C4:x'],
            '':   ['G3:C3:x'],
            'G2': ['G2:G1:C2:x']
        }
        self.assertEqual(param_owners, expected_owners)

        # verify vectors are set up correctly
        root._setup_vectors(param_owners, connections)

        expected_root_params   = ['G3:C3:x']
        expected_root_unknowns = ['G2:x', 'G2:G1:C2:y', 'G3:C3:y', 'G3:C4:y']

        expected_G3_params   = ['x', 'C3:x']
        expected_G3_unknowns = ['C3:y', 'C4:y']

        expected_G2_params   = ['x']
        expected_G2_unknowns = ['x', 'G1:C2:y']

        expected_G1_params   = ['x']
        expected_G1_unknowns = ['C2:y']

        self.assertEqual(list(root._varmanager.params.keys()),    expected_root_params)
        self.assertEqual(list(root._varmanager.dparams.keys()),   expected_root_params)
        self.assertEqual(list(root._varmanager.unknowns.keys()),  expected_root_unknowns)
        self.assertEqual(list(root._varmanager.dunknowns.keys()), expected_root_unknowns)
        self.assertEqual(list(root._varmanager.resids.keys()),    expected_root_unknowns)
        self.assertEqual(list(root._varmanager.dresids.keys()),   expected_root_unknowns)

        self.assertEqual(list(G3._varmanager.params.keys()),    expected_G3_params)
        self.assertEqual(list(G3._varmanager.dparams.keys()),   expected_G3_params)
        self.assertEqual(list(G3._varmanager.unknowns.keys()),  expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.dunknowns.keys()), expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.resids.keys()),    expected_G3_unknowns)
        self.assertEqual(list(G3._varmanager.dresids.keys()),   expected_G3_unknowns)

        self.assertEqual(list(G2._varmanager.params.keys()),    expected_G2_params)
        self.assertEqual(list(G2._varmanager.dparams.keys()),   expected_G2_params)
        self.assertEqual(list(G2._varmanager.unknowns.keys()),  expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.dunknowns.keys()), expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.resids.keys()),    expected_G2_unknowns)
        self.assertEqual(list(G2._varmanager.dresids.keys()),   expected_G2_unknowns)

        self.assertEqual(list(G1._varmanager.params.keys()),    expected_G1_params)
        self.assertEqual(list(G1._varmanager.dparams.keys()),   expected_G1_params)
        self.assertEqual(list(G1._varmanager.unknowns.keys()),  expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.dunknowns.keys()), expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.resids.keys()),    expected_G1_unknowns)
        self.assertEqual(list(G1._varmanager.dresids.keys()),   expected_G1_unknowns)

        # verify subsystem is using shared view of parent unknowns vector
        root._varmanager.unknowns['G2:x'] = 99.
        self.assertEqual(G2._varmanager.unknowns['x'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(root._varmanager.unknowns.metadata('G2:x'),
                         G2._varmanager.unknowns.metadata('x'))


if __name__ == "__main__":
    unittest.main()

""" Unit test for Groups."""

import unittest
from six import text_type, StringIO, itervalues
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem, Group, Relevance, IndepVarComp, ExecComp, ScipyGMRES, \
     Component
from openmdao.test.example_groups import ExampleGroup, ExampleGroupWithPromotes
from openmdao.test.simple_comps import SimpleImplicitComp


class TestGroup(unittest.TestCase):

    def test_add(self):

        group = Group()
        comp = ExecComp('y=x*2.0')
        group.add('mycomp', comp)

        subs = list(itervalues(group._subsystems))
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0], comp)
        self.assertEqual(subs[0].name, 'mycomp')

        comp2 = ExecComp('y=x*2.0')
        group.add("nextcomp", comp2)

        subs = list(itervalues(group._subsystems))
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0], comp)
        self.assertEqual(subs[1], comp2)

        with self.assertRaises(RuntimeError) as cm:
            group.add('mycomp', comp)

        expected_msg = "Group '' already contains a subsystem with name 'mycomp'."

        self.assertEqual(str(cm.exception), expected_msg)

    def test_bad_sysname(self):
        group = Group()
        try:
            group.add('0', ExecComp('y=x*2.0'), promotes=['x'])
        except NameError as err:
            self.assertEqual(str(err), ": '0' is not a valid system name.")
        try:
            group.add('foo:bar', ExecComp('y=x*2.0'), promotes=['x'])
        except NameError as err:
            self.assertEqual(str(err), ": 'foo:bar' is not a valid system name.")

    def test_variables(self):
        group = Group()
        group.add('C1', ExecComp('y=x*2.0'), promotes=['x'])
        group.add("C2", ExecComp('y=x*2.0'), promotes=['y'])

        prob = Problem(root=group)
        prob.setup(check=False)
        params_dict = prob.root._params_dict
        unknowns_dict = prob.root._unknowns_dict

        self.assertEqual(list(params_dict.keys()), ['C1.x', 'C2.x'])
        self.assertEqual(list(unknowns_dict.keys()), ['C1.y', 'C2.y'])

        to_prom_name = prob.root._sysdata.to_prom_name
        self.assertEqual([to_prom_name[n] for n in params_dict], ['x', 'C2.x'])
        self.assertEqual([to_prom_name[n] for n in unknowns_dict], ['C1.y', 'y'])

    def test_multiple_connect(self):
        root = Group()
        C1 = root.add('C1', ExecComp('y=x*2.0'))
        C2 = root.add('C2', ExecComp('y=x*2.0'))
        C3 = root.add('C3', ExecComp('y=x*2.0'))

        root.connect('C1.y',['C2.x', 'C3.x'])

        prob = Problem()
        root._init_sys_data('', prob._probdata)
        params_dict, unknowns_dict = root._setup_variables()

        # verify we get correct connection information
        connections = root._get_explicit_connections()
        expected_connections = {
            'C2.x': [('C1.y', None)],
            'C3.x': [('C1.y', None)]
        }
        self.assertEqual(connections, expected_connections)

    def test_connect(self):
        root = ExampleGroup()
        prob = Problem(root=root)

        prob.setup(check=False)

        self.assertEqual(root.pathname, '')
        self.assertEqual(root.G3.pathname, 'G3')
        self.assertEqual(root.G2.pathname, 'G2')
        self.assertEqual(root.G1.pathname, 'G2.G1')

        # verify variables are set up correctly
        #root._setup_variables()

        # TODO: check for expected results from _setup_variables
        self.assertEqual(list(root.G1._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.G1.C2.x', 'size': 1})])
        self.assertEqual(list(root.G1._unknowns_dict.items()),
                         [('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y',
                                          'size': 1})])

        self.assertEqual(list(root.G2._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.G1.C2.x',
                                          'size': 1})])
        self.assertEqual(list(root.G2._unknowns_dict.items()),
                         [('G2.C1.x', {'shape': 1, 'pathname': 'G2.C1.x', 'val': 5.0,
                                       'top_promoted_name': 'G2.C1.x', 'size': 1}),
                          ('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y',
                                          'size': 1})])

        self.assertEqual(list(root.G3._params_dict.items()),
                         [('G3.C3.x', {'shape': 1, 'pathname': 'G3.C3.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C3.x', 'size': 1}),
                          ('G3.C4.x', {'shape': 1, 'pathname': 'G3.C4.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C4.x', 'size': 1})])
        self.assertEqual(list(root.G3._unknowns_dict.items()),
                         [('G3.C3.y', {'shape': 1, 'pathname': 'G3.C3.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C3.y', 'size': 1}),
                          ('G3.C4.y', {'shape': 1, 'pathname': 'G3.C4.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C4.y', 'size': 1})])

        self.assertEqual(list(root._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.G1.C2.x', 'size': 1}),
                          ('G3.C3.x', {'shape': 1, 'pathname': 'G3.C3.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C3.x', 'size': 1}),
                          ('G3.C4.x', {'shape': 1, 'pathname': 'G3.C4.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C4.x', 'size': 1})])
        self.assertEqual(list(root._unknowns_dict.items()),
                         [('G2.C1.x', {'shape': 1, 'pathname': 'G2.C1.x', 'val': 5.0,
                                       'top_promoted_name': 'G2.C1.x', 'size': 1}),
                          ('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y',
                                          'size': 1}),
                          ('G3.C3.y', {'shape': 1, 'pathname': 'G3.C3.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C3.y', 'size': 1}),
                          ('G3.C4.y', {'shape': 1, 'pathname': 'G3.C4.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C4.y', 'size': 1})])

        # verify we get correct connection information
        #connections = root._get_explicit_connections()
        expected_connections = {
            'G2.G1.C2.x': ('G2.C1.x', None),
            'G3.C3.x':    ('G2.G1.C2.y', None),
            'G3.C4.x':    ('G3.C3.y', None)
        }
        self.assertEqual(root.connections, expected_connections)

        expected_root_params   = ['G3.C3.x']
        expected_root_unknowns = ['G2.C1.x', 'G2.G1.C2.y', 'G3.C3.y', 'G3.C4.y']

        expected_G3_params   = ['C4.x', 'C3.x']
        expected_G3_unknowns = ['C3.y', 'C4.y']

        expected_G2_params   = ['G1.C2.x']
        expected_G2_unknowns = ['C1.x', 'G1.C2.y']

        expected_G1_params   = ['C2.x']
        expected_G1_unknowns = ['C2.y']

        voi = None

        self.assertEqual(list(root.params.keys()),    expected_root_params)
        self.assertEqual(list(root.dpmat[voi].keys()),   expected_root_params)
        self.assertEqual(list(root.unknowns.keys()),  expected_root_unknowns)
        self.assertEqual(list(root.dumat[voi].keys()), expected_root_unknowns)
        self.assertEqual(list(root.resids.keys()),    expected_root_unknowns)
        self.assertEqual(list(root.drmat[voi].keys()),   expected_root_unknowns)

        self.assertEqual(list(root.G3.params.keys()),    expected_G3_params)
        self.assertEqual(list(root.G3.dpmat[voi].keys()),   expected_G3_params)
        self.assertEqual(list(root.G3.unknowns.keys()),  expected_G3_unknowns)
        self.assertEqual(list(root.G3.dumat[voi].keys()), expected_G3_unknowns)
        self.assertEqual(list(root.G3.resids.keys()),    expected_G3_unknowns)
        self.assertEqual(list(root.G3.drmat[voi].keys()),   expected_G3_unknowns)

        self.assertEqual(list(root.G2.params.keys()),    expected_G2_params)
        self.assertEqual(list(root.G2.dpmat[voi].keys()),   expected_G2_params)
        self.assertEqual(list(root.G2.unknowns.keys()),  expected_G2_unknowns)
        self.assertEqual(list(root.G2.dumat[voi].keys()), expected_G2_unknowns)
        self.assertEqual(list(root.G2.resids.keys()),    expected_G2_unknowns)
        self.assertEqual(list(root.G2.drmat[voi].keys()),   expected_G2_unknowns)

        self.assertEqual(list(root.G1.params.keys()),    expected_G1_params)
        self.assertEqual(list(root.G1.dpmat[voi].keys()),   expected_G1_params)
        self.assertEqual(list(root.G1.unknowns.keys()),  expected_G1_unknowns)
        self.assertEqual(list(root.G1.dumat[voi].keys()), expected_G1_unknowns)
        self.assertEqual(list(root.G1.resids.keys()),    expected_G1_unknowns)
        self.assertEqual(list(root.G1.drmat[voi].keys()),   expected_G1_unknowns)

        # verify subsystem is using shared view of parent unknowns vector
        root.unknowns['G2.C1.x'] = 99.
        self.assertEqual(root.G2.unknowns['C1.x'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(root.unknowns.metadata('G2.C1.x'),
                         root.G2.unknowns.metadata('C1.x'))

    def test_promotes(self):
        root = ExampleGroupWithPromotes()
        prob = Problem(root=root)

        prob.setup(check=False)

        self.assertEqual(root.pathname, '')
        self.assertEqual(root.G3.pathname, 'G3')
        self.assertEqual(root.G2.pathname, 'G2')
        self.assertEqual(root.G1.pathname, 'G2.G1')

        # TODO: check for expected results from _setup_variables
        self.assertEqual(list(root.G1._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.x', 'size': 1})])
        self.assertEqual(list(root.G1._unknowns_dict.items()),
                         [('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y', 'size': 1})])

        self.assertEqual(list(root.G2._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.x', 'size': 1})])
        self.assertEqual(list(root.G2._unknowns_dict.items()),
                         [('G2.C1.x', {'shape': 1, 'pathname': 'G2.C1.x', 'val': 5.0,
                                       'top_promoted_name': 'G2.x', 'size': 1}),
                          ('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y', 'size': 1})])

        self.assertEqual(list(root.G3._params_dict.items()),
                         [('G3.C3.x', {'shape': 1, 'pathname': 'G3.C3.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C3.x', 'size': 1}),
                          ('G3.C4.x', {'shape': 1, 'pathname': 'G3.C4.x', 'val': 3.0,
                                       'top_promoted_name': 'x', 'size': 1})])

        self.assertEqual(list(root.G3._unknowns_dict.items()),
                         [('G3.C3.y', {'shape': 1, 'pathname': 'G3.C3.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C3.y', 'size': 1}),
                          ('G3.C4.y', {'shape': 1, 'pathname': 'G3.C4.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C4.y', 'size': 1})])

        self.assertEqual(list(root._params_dict.items()),
                         [('G2.G1.C2.x', {'shape': 1, 'pathname': 'G2.G1.C2.x', 'val': 3.0,
                                          'top_promoted_name': 'G2.x', 'size': 1}),
                          ('G3.C3.x', {'shape': 1, 'pathname': 'G3.C3.x', 'val': 3.0,
                                       'top_promoted_name': 'G3.C3.x', 'size': 1}),
                          ('G3.C4.x', {'shape': 1, 'pathname': 'G3.C4.x', 'val': 3.0,
                                       'top_promoted_name': 'x', 'size': 1})])

        self.assertEqual(list(root._unknowns_dict.items()),
                         [('G2.C1.x', {'shape': 1, 'pathname': 'G2.C1.x', 'val': 5.0,
                                       'top_promoted_name': 'G2.x', 'size': 1}),
                          ('G2.G1.C2.y', {'shape': 1, 'pathname': 'G2.G1.C2.y', 'val': 5.5,
                                          'top_promoted_name': 'G2.G1.C2.y',
                                          'size': 1}),
                          ('G3.C3.y', {'shape': 1, 'pathname': 'G3.C3.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C3.y', 'size': 1}),
                          ('G3.C4.y', {'shape': 1, 'pathname': 'G3.C4.y', 'val': 5.5,
                                       'top_promoted_name': 'G3.C4.y', 'size': 1})])

        expected_root_params   = ['G3.C3.x']
        expected_root_unknowns = ['G2.x', 'G2.G1.C2.y', 'G3.C3.y', 'G3.C4.y']

        expected_G3_params   = ['C4.x', 'C3.x']
        expected_G3_unknowns = ['C3.y', 'C4.y']

        expected_G2_params   = ['G1.C2.x']
        expected_G2_unknowns = ['x', 'G1.C2.y']

        expected_G1_params   = ['C2.x']
        expected_G1_unknowns = ['C2.y']

        voi = None

        self.assertEqual(list(root.params.keys()),    expected_root_params)
        self.assertEqual(list(root.dpmat[voi].keys()),   expected_root_params)
        self.assertEqual(list(root.unknowns.keys()),  expected_root_unknowns)
        self.assertEqual(list(root.dumat[voi].keys()), expected_root_unknowns)
        self.assertEqual(list(root.resids.keys()),    expected_root_unknowns)
        self.assertEqual(list(root.drmat[voi].keys()),   expected_root_unknowns)

        self.assertEqual(list(root.G3.params.keys()),    expected_G3_params)
        self.assertEqual(list(root.G3.dpmat[voi].keys()),   expected_G3_params)
        self.assertEqual(list(root.G3.unknowns.keys()),  expected_G3_unknowns)
        self.assertEqual(list(root.G3.dumat[voi].keys()), expected_G3_unknowns)
        self.assertEqual(list(root.G3.resids.keys()),    expected_G3_unknowns)
        self.assertEqual(list(root.G3.drmat[voi].keys()),   expected_G3_unknowns)

        self.assertEqual(list(root.G2.params.keys()),    expected_G2_params)
        self.assertEqual(list(root.G2.dpmat[voi].keys()),   expected_G2_params)
        self.assertEqual(list(root.G2.unknowns.keys()),  expected_G2_unknowns)
        self.assertEqual(list(root.G2.dumat[voi].keys()), expected_G2_unknowns)
        self.assertEqual(list(root.G2.resids.keys()),    expected_G2_unknowns)
        self.assertEqual(list(root.G2.drmat[voi].keys()),   expected_G2_unknowns)

        self.assertEqual(list(root.G1.params.keys()),    expected_G1_params)
        self.assertEqual(list(root.G1.dpmat[voi].keys()),   expected_G1_params)
        self.assertEqual(list(root.G1.unknowns.keys()),  expected_G1_unknowns)
        self.assertEqual(list(root.G1.dumat[voi].keys()), expected_G1_unknowns)
        self.assertEqual(list(root.G1.resids.keys()),    expected_G1_unknowns)
        self.assertEqual(list(root.G1.drmat[voi].keys()),   expected_G1_unknowns)

        # verify subsystem is using shared view of parent unknowns vector
        root.unknowns['G2.x'] = 99.
        self.assertEqual(root.G2.unknowns['x'], 99.)

        # verify subsystem is getting correct metadata from parent unknowns vector
        self.assertEqual(root.unknowns.metadata('G2.x'),
                         root.G2.unknowns.metadata('x'))

    def test_variable_access(self):
        prob = Problem(root=ExampleGroup())

        # try accessing variable value before setup()
        try:
            prob['G2.C1.x']
        except Exception as err:
            msg = "'unknowns' has not been initialized, setup() must be called before 'G2.C1.x' can be accessed"
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

        prob.setup(check=False)

        # check that we can access values from unknowns (default) and params
        self.assertEqual(prob['G2.C1.x'], 5.)             # default output from IndepVarComp
        self.assertEqual(prob['G2.G1.C2.y'], 5.5)         # output from ExecComp
        self.assertEqual(prob.root.G3.C3.params['x'], 0.)      # initial value for a parameter
        self.assertEqual(prob.root.G2.G1.C2.params['x'], 0.)   # initial value for a parameter

        # now try same thing in a Group with promotes
        prob = Problem(root=ExampleGroupWithPromotes())
        prob.setup(check=False)

        prob.root.G2.unknowns['x'] = 99.

        self.assertEqual(prob['G2.x'],   99.)
        self.assertEqual(prob.root.G2.G1.C2.params['x'], 0.)   # initial value for a parameter

        # and make sure we get the correct value after a transfer
        prob.root.G2._transfer_data('G1')
        self.assertEqual(prob.root.G2.G1.C2.params['x'], 99.)  # transferred value of parameter

    def test_subsystem_access(self):
        prob = Problem(root=ExampleGroup())
        root = prob.root

        self.assertEqual(root.G2, prob.root.G2)
        self.assertEqual(root.G2.C1, prob.root.C1)
        self.assertEqual(root.G2.G1, prob.root.G1)
        self.assertEqual(root.G2.G1.C2, prob.root.C2)

        self.assertEqual(root.G3, prob.root.G3)
        self.assertEqual(root.G3.C3, prob.root.C3)
        self.assertEqual(root.G3.C4, prob.root.C4)

        prob = Problem(root=ExampleGroupWithPromotes())
        root = prob.root

        self.assertEqual(root.G2, prob.root.G2)
        self.assertEqual(root.G2.C1, prob.root.C1)
        self.assertEqual(root.G2.G1, prob.root.G1)
        self.assertEqual(root.G2.G1.C2, prob.root.C2)

        self.assertEqual(root.G3, prob.root.G3)
        self.assertEqual(root.G3.C3, prob.root.C3)
        self.assertEqual(root.G3.C4, prob.root.C4)

    def test_nested_conn(self):
        # tests a self contained connection under a Group nested at least 3 levels
        # down from the Problem
        prob = Problem(root=Group())
        root = prob.root
        G1 = root.add('G1', Group())
        G2 = G1.add('G2', Group())
        C1 = G2.add('C1', IndepVarComp('x', 5.))
        C2 = G2.add('C2', ExecComp('y=x*2.0'))
        G2.connect('C1.x', 'C2.x')
        prob.setup(check=False)

    def test_fd_params(self):
        # tests retrieval of a list of any internal params whose source is either
        # a IndepVarComp or is outside of the Group
        prob = Problem(root=ExampleGroup())
        prob.setup(check=False)
        root = prob.root

        self.assertEqual(root._get_fd_params(), ['G2.G1.C2.x'])
        self.assertEqual(root.G2._get_fd_params(), ['G1.C2.x'])
        self.assertEqual(root.G2.G1._get_fd_params(), ['C2.x'])
        self.assertEqual(root.G3._get_fd_params(), ['C3.x'])

        self.assertEqual(root.G3.C3._get_fd_params(), ['x'])
        self.assertEqual(root.G2.G1.C2._get_fd_params(), ['x'])

    def test_fd_unknowns(self):
        # tests retrieval of a list of any internal unknowns with IndepVarComp
        # variables filtered out.
        prob = Problem(root=ExampleGroup())
        prob.setup(check=False)
        root = prob.root

        self.assertEqual(root._get_fd_unknowns(), ['G2.G1.C2.y', 'G3.C3.y', 'G3.C4.y'])
        self.assertEqual(root.G2._get_fd_unknowns(), ['G1.C2.y'])
        self.assertEqual(root.G2.G1._get_fd_unknowns(), ['C2.y'])
        self.assertEqual(root.G3._get_fd_unknowns(), ['C3.y', 'C4.y'])

        self.assertEqual(root.G3.C3._get_fd_unknowns(), ['y'])
        self.assertEqual(root.G2.G1.C2._get_fd_unknowns(), ['y'])

    def test_dump(self):
        prob = Problem(root=ExampleGroup())
        prob.setup(check=False)
        save = StringIO()
        prob.root.dump(out_stream=save)

        # don't want to write a test that does a string compare of a dump, so
        # for now, just verify that calling dump doesn't raise an exception.

    def test_data_xfer(self):
        prob = Problem(root=ExampleGroupWithPromotes())
        prob.setup(check=False)

        prob.root.unknowns['G2.G1.C2.y'] = 99.
        self.assertEqual(prob['G2.G1.C2.y'], 99.)

        prob.root._transfer_data('G3')
        self.assertEqual(prob.root.params['G3.C3.x'], 99.)

        self.assertEqual(prob['G3.C3.x'], 99.)

    def test_list_and_set_order(self):

        prob = Problem(root=ExampleGroupWithPromotes())

        order1 = prob.root.list_order()
        self.assertEqual(order1, ['G2', 'G3'])

        # Big boy rules
        order2 = ['G3', 'G2']

        prob.root.set_order(order2)

        order1 = prob.root.list_order()
        self.assertEqual(order1, ['G3', 'G2'])

        # Extra
        order2 = ['G3', 'G2', 'junk']
        with self.assertRaises(ValueError) as cm:
            prob.root.set_order(order2)

        msg = "Unexpected new order. "
        msg += "The following are extra: ['junk']. "
        self.assertEqual(str(cm.exception), msg)

        # Missing
        order2 = ['G3']
        with self.assertRaises(ValueError) as cm:
            prob.root.set_order(order2)

        msg = "Unexpected new order. "
        msg += "The following are missing: ['G2']. "
        self.assertEqual(str(cm.exception), msg)

        # Extra and Missing
        order2 = ['G3', 'junk']
        with self.assertRaises(ValueError) as cm:
            prob.root.set_order(order2)

        msg = "Unexpected new order. "
        msg += "The following are missing: ['G2']. "
        msg += "The following are extra: ['junk']. "
        self.assertEqual(str(cm.exception), msg)

        # Dupes
        order2 = ['G3', 'G2', 'G3']
        with self.assertRaises(ValueError) as cm:
            prob.root.set_order(order2)

        msg = "Duplicate name(s) found in order list: ['G3']"
        self.assertEqual(str(cm.exception), msg)

        # Don't let user call add.
        with self.assertRaises(RuntimeError) as cm:
            prob.root.add('C5', Group())

        msg = 'You cannot call add after specifying an order.'
        self.assertEqual(str(cm.exception), msg)

    def test_auto_order(self):
        # this tests the auto ordering when we have a cycle that is smaller
        # than the full graph.
        p = Problem(root=Group())
        root = p.root
        root.ln_solver = ScipyGMRES()

        C5 = root.add("C5", ExecComp('y=x*2.0'))
        C6 = root.add("C6", ExecComp('y=x*2.0'))
        C1 = root.add("C1", ExecComp('y=x*2.0'))
        C2 = root.add("C2", ExecComp('y=x*2.0'))
        C3 = root.add("C3", ExecComp(['y=x*2.0','y2=x2+1.0']))
        C4 = root.add("C4", ExecComp(['y=x*2.0','y2=x2+1.0']))
        P1 = root.add("P1", IndepVarComp('x', 1.0))

        root.connect('P1.x', 'C1.x')
        root.connect('C1.y', 'C2.x')
        root.connect('C2.y', 'C4.x')
        root.connect('C4.y', 'C5.x')
        root.connect('C5.y', 'C6.x')
        root.connect('C5.y', 'C3.x2')
        root.connect('C6.y', 'C3.x')
        root.connect('C3.y', 'C4.x2')

        p.setup(check=False)

        self.assertEqual(p.root.list_auto_order()[0],
                         ['P1','C1','C2','C4','C5','C6','C3'])

    def test_auto_order2(self):
        # this tests the auto ordering when we have a cycle that is the full graph.
        p = Problem(root=Group())
        root = p.root
        root.ln_solver = ScipyGMRES()
        C1 = root.add("C1", ExecComp('y=x*2.0'))
        C2 = root.add("C2", ExecComp('y=x*2.0'))
        C3 = root.add("C3", ExecComp('y=x*2.0'))

        root.connect('C1.y', 'C3.x')
        root.connect('C3.y', 'C2.x')
        root.connect('C2.y', 'C1.x')

        p.setup(check=False)

        self.assertEqual(p.root.list_auto_order()[0], ['C1', 'C3', 'C2'])

    def test_list_states(self):

        top = Problem()
        root = top.root = Group()
        sub = root.add('sub', Group())
        sub.add('comp', SimpleImplicitComp())
        sub.ln_solver = ScipyGMRES()
        top.setup(check=False)
        top['sub.comp.z'] = 7.7

        stream = cStringIO()
        root.list_states(stream=stream)
        self.assertTrue('sub.comp.z' in stream.getvalue())
        self.assertTrue('Value: 7.7' in stream.getvalue())
        self.assertTrue('Residual: 0.0' in stream.getvalue())
        self.assertTrue('States in model:' in stream.getvalue())

        stream = cStringIO()
        sub.list_states(stream=stream)
        self.assertTrue('comp.z' in stream.getvalue())
        self.assertTrue('Value: 7.7' in stream.getvalue())
        self.assertTrue('Residual: 0.0' in stream.getvalue())
        self.assertTrue('sub.comp.z' not in stream.getvalue())
        self.assertTrue('States in sub:' in stream.getvalue())

        top = Problem()
        root = top.root = ExampleGroupWithPromotes()
        top.setup(check=False)

        stream = cStringIO()
        root.list_states(stream=stream)
        self.assertTrue('No states in model.' in stream.getvalue())

        stream = cStringIO()
        root.G2.list_states(stream=stream)
        self.assertTrue('No states in G2.' in stream.getvalue())

        class ArrayImplicitComp(Component):
            """ Needed to test arrays here. """

            def __init__(self):
                super(ArrayImplicitComp, self).__init__()

                # Params
                self.add_param('x', np.zeros((3, 1)))

                # Unknowns
                self.add_output('y', np.zeros((3, 1)))

                # States
                self.add_state('z', 2.0*np.ones((3, 1)), lower=1.5, upper=np.array([2.5, 2.6, 2.7]))

            def solve_nonlinear(self, params, unknowns, resids):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""
                pass

            def linearize(self, params, unknowns, resids):
                """Analytical derivatives."""
                pass

        top = Problem()
        root = top.root = Group()
        root.add('comp', ArrayImplicitComp())
        root.ln_solver.options['maxiter'] = 2
        top.setup(check=False)

        stream = cStringIO()
        root.list_states(stream=stream)
        base = 'States in model:\n\ncomp.z\nValue: [[ 2.]\n [ 2.]\n [ 2.]]\nResidual: [[ 0.]\n [ 0.]\n [ 0.]]'
        self.assertTrue(base in stream.getvalue())

    def test_list_params(self):

        top = Problem()
        root = top.root = Group()
        g1 = root.add('g1', Group(), promotes=['b', 'f'])
        g2 = g1.add('g2', Group(), promotes=['c', 'e'])

        root.add('comp1', ExecComp(['b = a']), promotes=['b'])
        g1.add('comp2', ExecComp(['c = b + p1']), promotes=['b', 'c'])
        g1.add('comp3', ExecComp(['c_a = b_a + p2']))
        g2.add('comp4', ExecComp(['d = c + p3']), promotes=['c', 'd'])
        g2.add('comp5', ExecComp(['d_a = c_a + p4']))
        g2.add('comp6', ExecComp(['e = d + p5']), promotes=['d', 'e'])
        g2.add('comp7', ExecComp(['e_a = d_a + p6']))
        g1.add('comp8', ExecComp(['f = e + p7']), promotes=['f', 'e'])
        g1.add('comp9', ExecComp(['f_a = e_a + p8']))
        root.add('comp10', ExecComp(['g = f + p9']), promotes=['f'])
        root.add('comp11', ExecComp(['g_a = f_a + p10']))

        root.connect('b', 'g1.comp3.b_a')
        root.connect('g1.comp3.c_a', 'g1.g2.comp5.c_a')
        root.connect('g1.g2.comp5.d_a', 'g1.g2.comp7.d_a')
        root.connect('g1.g2.comp7.e_a', 'g1.comp9.e_a')
        root.connect('g1.comp9.f_a', 'comp11.f_a')

        root.add('p1', IndepVarComp('a', 1.0))
        root.connect('p1.a', 'comp1.a')

        top.setup(check=False)

        plist1, plist2 = g2.list_params(stream=None)

        self.assertEqual(plist1, ['g1.g2.comp4.p3', 'g1.g2.comp5.p4', 'g1.g2.comp6.p5', 'g1.g2.comp7.p6'])
        self.assertEqual(plist2, ['g1.g2.comp4.c', 'g1.g2.comp5.c_a'])

        plist1, plist2 = g1.list_params(stream=None)

        self.assertEqual(plist1, ['g1.g2.comp4.p3', 'g1.g2.comp5.p4', 'g1.g2.comp6.p5', 'g1.g2.comp7.p6', 'g1.comp2.p1', 'g1.comp3.p2', 'g1.comp8.p7', 'g1.comp9.p8'])
        self.assertEqual(plist2, ['g1.comp2.b', 'g1.comp3.b_a'])

if __name__ == "__main__":
    unittest.main()

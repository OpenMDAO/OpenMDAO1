import unittest
import numpy as np
from six import text_type, PY3
from six.moves import cStringIO
import warnings

from openmdao.api import Problem, Group, IndepVarComp, ExecComp


class TestConnections(unittest.TestCase):

    def setUp(self):
        self.p = Problem(root=Group())
        root = self.p.root

        self.G1 = root.add("G1", Group())
        self.G2 = self.G1.add("G2", Group())
        self.C1 = self.G2.add("C1", ExecComp('y=x*2.0'))
        self.C2 = self.G2.add("C2", IndepVarComp('x', 1.0))

        self.G3 = root.add("G3", Group())
        self.G4 = self.G3.add("G4", Group())
        self.C3 = self.G4.add("C3", ExecComp('y=x*2.0'))
        self.C4 = self.G4.add("C4", ExecComp('y=x*2.0'))

    def test_diff_conn_input_vals(self):
        # set different initial values
        self.C1._params_dict['x']['val'] = 7.
        self.C3._params_dict['x']['val'] = 5.

        # connect two inputs
        self.p.root.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            self.p.setup(check=False)

            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message),
                    "The following sourceless connected inputs have different initial values: "
                    "[('G1.G2.C1.x', 7.0), ('G3.G4.C3.x', 5.0)].")

    def test_diff_conn_input_units(self):
        # set different but compatible units
        self.C1._params_dict['x']['units'] = 'ft'
        self.C3._params_dict['x']['units'] = 'in'

        # connect two inputs
        self.p.root.connect('G1.G2.C1.x', 'G3.G4.C3.x')

        try:
            self.p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "The following connected inputs have no source in "
                             "unknowns but their units differ: "
                             "[('G1.G2.C1.x', 'ft'), ('G3.G4.C3.x', 'in')]")

    def test_no_conns(self):
        self.p.setup(check=False)
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(self.p._dangling['G3.G4.C3.x'], set(['G3.G4.C3.x']))
        self.assertEqual(self.p._dangling['G3.G4.C4.x'], set(['G3.G4.C4.x']))
        self.assertEqual(len(self.p._dangling), 3)

        self.p['G1.G2.C1.x'] = 111.
        self.p['G3.G4.C3.x'] = 222.
        self.p['G3.G4.C4.x'] = 333.

        self.assertEqual(self.p.root.G1.G2.C1.params['x'], 111.)
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 222.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 333.)

    def test_inp_inp_conn_no_src(self):
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x')

        stream = cStringIO()
        self.p.setup(out_stream=stream)
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(self.p._dangling['G3.G4.C3.x'], set(['G3.G4.C3.x', 'G3.G4.C4.x']))
        self.assertEqual(self.p._dangling['G3.G4.C4.x'], set(['G3.G4.C4.x', 'G3.G4.C3.x']))
        self.assertEqual(len(self.p._dangling), 3)

        self.p['G3.G4.C3.x'] = 999.
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 999.)

        content = stream.getvalue()
        self.assertTrue("The following parameters have no associated unknowns:\nG1.G2.C1.x\nG3.G4.C3.x\nG3.G4.C4.x" in content)
        self.assertTrue("The following components have no connections:\nG1.G2.C1\nG1.G2.C2\nG3.G4.C3\nG3.G4.C4\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

    def test_inp_inp_conn_w_src(self):
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x')
        self.p.root.connect('G1.G2.C2.x', 'G3.G4.C3.x')
        self.p.setup(check=False)
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(len(self.p._dangling), 1)

        self.p['G1.G2.C2.x'] = 999.
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 0.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 0.)

        self.p.run()
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 999.)

    def test_inp_inp_conn_w_src2(self):
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x')
        self.p.root.connect('G1.G2.C2.x', 'G3.G4.C4.x')
        self.p.setup(check=False)
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(len(self.p._dangling), 1)

        self.p['G1.G2.C2.x'] = 999.
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 0.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 0.)

        self.p.run()
        self.assertEqual(self.p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(self.p.root.G3.G4.C4.params['x'], 999.)


class TestConnectionsPromoted(unittest.TestCase):

    def test_inp_inp_promoted_no_src(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group())
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", ExecComp('y=x*2.0'))

        G3 = root.add("G3", Group())
        G4 = G3.add("G4", Group())
        C3 = G4.add("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add("C4", ExecComp('y=x*2.0'), promotes=['x'])

        stream = cStringIO()
        checks = p.setup(out_stream=stream)
        self.assertEqual(checks['dangling_params'],
                         ['G1.G2.C1.x', 'G1.G2.C2.x', 'G3.G4.x'])
        self.assertEqual(checks['no_connect_comps'],
                         ['G1.G2.C1', 'G1.G2.C2', 'G3.G4.C3', 'G3.G4.C4'])
        self.assertEqual(p._dangling['G3.G4.x'], set(['G3.G4.C3.x','G3.G4.C4.x']))

        # setting promoted name should set both params mapped to that name since the
        # params are dangling.
        p['G3.G4.x'] = 999.
        self.assertEqual(p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 999.)

    def test_inp_inp_promoted_no_src2(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group())
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", ExecComp('y=x*2.0'))

        G3 = root.add("G3", Group())
        G4 = G3.add("G4", Group(), promotes=['x'])
        C3 = G4.add("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup(check=False)

        self.assertEqual(p._dangling['G3.x'], set(['G3.G4.C3.x','G3.G4.C4.x']))

        # setting promoted name should set both params mapped to that name
        p['G3.x'] = 999.
        self.assertEqual(p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 999.)

    def test_inp_inp_promoted_w_prom_src(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add("G1", Group(), promotes=['x'])
        G2 = G1.add("G2", Group(), promotes=['x'])
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add("G3", Group(), promotes=['x'])
        G4 = G3.add("G4", Group(), promotes=['x'])
        C3 = G4.add("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.setup(check=False)

        self.assertEqual(p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(len(p._dangling), 1)

        # setting promoted name will set the value into the unknowns, but will
        # not propagate it to the params. That will happen during run().
        p['x'] = 999.
        self.assertEqual(p.root.G3.G4.C3.params['x'], 0.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 0.)

        p.run()
        self.assertEqual(p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 999.)

    def test_inp_inp_promoted_w_explicit_src(self):
        p = Problem(root=Group())
        root = p.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group(), promotes=['x'])
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", IndepVarComp('x', 1.0), promotes=['x'])

        G3 = root.add("G3", Group())
        G4 = G3.add("G4", Group(), promotes=['x'])
        C3 = G4.add("C3", ExecComp('y=x*2.0'), promotes=['x'])
        C4 = G4.add("C4", ExecComp('y=x*2.0'), promotes=['x'])

        p.root.connect('G1.x', 'G3.x')
        p.setup(check=False)

        self.assertEqual(p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(len(p._dangling), 1)

        # setting promoted name will set the value into the unknowns, but will
        # not propagate it to the params. That will happen during run().
        p['G1.x'] = 999.
        self.assertEqual(p.root.G3.G4.C3.params['x'], 0.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 0.)

        p.run()
        self.assertEqual(p.root.G3.G4.C3.params['x'], 999.)
        self.assertEqual(p.root.G3.G4.C4.params['x'], 999.)



if __name__ == "__main__":
    unittest.main()

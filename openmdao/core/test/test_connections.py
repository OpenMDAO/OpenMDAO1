import unittest
import numpy as np
from six import text_type, PY3
import warnings

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp


class TestConnections(unittest.TestCase):

    def setUp(self):
        self.p = Problem(root=Group())
        root = self.p.root

        self.G1 = root.add("G1", Group())
        self.G2 = self.G1.add("G2", Group())
        self.C1 = self.G2.add("C1", ExecComp('y=x*2.0'))
        self.C2 = self.G2.add("C2", ExecComp('y=x*2.0'))

        self.G3 = root.add("G3", Group())
        self.G4 = self.G3.add("G4", Group())
        self.C3 = self.G4.add("C3", ExecComp('y=x*2.0'))
        self.C4 = self.G4.add("C4", ExecComp('y=x*2.0'))

    def test_no_conns(self):
        self.p.setup()
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(self.p._dangling['G1.G2.C2.x'], set(['G1.G2.C2.x']))
        self.assertEqual(self.p._dangling['G3.G4.C3.x'], set(['G3.G4.C3.x']))
        self.assertEqual(self.p._dangling['G3.G4.C4.x'], set(['G3.G4.C4.x']))
        self.assertEqual(len(self.p._dangling), 4)

    def test_inp_inp_conn_no_src(self):
        self.p.root.connect('G3.G4.C3.x', 'G3.G4.C4.x')
        self.p.setup()
        self.assertEqual(self.p._dangling['G1.G2.C1.x'], set(['G1.G2.C1.x']))
        self.assertEqual(self.p._dangling['G1.G2.C2.x'], set(['G1.G2.C2.x']))
        self.assertEqual(self.p._dangling['G3.G4.C3.x'], set(['G3.G4.C3.x', 'G3.G4.C4.x']))
        self.assertEqual(self.p._dangling['G3.G4.C4.x'], set(['G3.G4.C4.x', 'G3.G4.C3.x']))
        self.assertEqual(len(self.p._dangling), 4)



if __name__ == "__main__":
    unittest.main()

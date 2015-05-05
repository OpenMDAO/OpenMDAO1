import unittest
from collections import OrderedDict

from openmdao.core.varmanager import VarManager
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp

class TestVarManager(unittest.TestCase):

    def test_scatter(self):
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

        prob = Problem(root)
        prob.setup()

        root._varmanager.unknowns['G2:G1:C2:y'] = 99.

        root._varmanager._scatter('G3')
        self.assertEqual(root._varmanager.params['G3:C3:x'], 99.)


if __name__ == "__main__":
    unittest.main()

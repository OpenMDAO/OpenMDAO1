import unittest
from collections import OrderedDict

from openmdao.core.varmanager import VarManager
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp
from openmdao.test.examplegroups import ExampleGroupWithPromotes

class TestVarManager(unittest.TestCase):

    def test_data_xfer(self):
        root = ExampleGroupWithPromotes()

        prob = Problem(root)
        prob.setup()

        root.unknowns['G2:G1:C2:y'] = 99.
        self.assertEqual(root['G2:G1:C2:y'], 99.)

        root._varmanager._transfer_data('G3')
        self.assertEqual(root.params['G3:C3:x'], 99.)

        self.assertEqual(root['G3:C3:x', 'params'], 99.)


if __name__ == "__main__":
    unittest.main()

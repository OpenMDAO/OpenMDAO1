import unittest
import numpy as np
from collections import OrderedDict

from openmdao.core.varmanager import VarManager

class TestVarManager(unittest.TestCase):

    def test_varmanager(self):
        params = OrderedDict()
        unknowns = OrderedDict()
        states = OrderedDict()
        #
        # params['x1'] = {}
        # params['x2'] = {}
        # unknowns['y1'] = { 'val': np.ones(3) }
        # states['y2'] = { 'val': 2.0 }
        #
        # vm = ProblemVarManager(params, unknowns, states)
        #
        # self.assertEqual(vm.params), 2)
        # self.assertEqual(len)
        #
        # self.assertEqual(vm.params['x'], 1.0)
        # self.assertEqual(vm.unknowns['y'])


if __name__ == "__main__":
    unittest.main()

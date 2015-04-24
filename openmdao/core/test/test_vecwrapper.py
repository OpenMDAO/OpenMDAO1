import unittest
import numpy as np
from collections import OrderedDict

from openmdao.core.vecwrapper import SourceVecWrapper, TargetVecWrapper

class TestVecWrapper(unittest.TestCase):

    def test_vecwrapper(self):

        unknowns = OrderedDict()
        states = OrderedDict()

        unknowns['y1'] = { 'val': np.ones((3, 2)) }
        unknowns['y2'] = { 'val': 2.0 }
        unknowns['y3'] = { 'val': "foo" }
        states['s1'] = { 'val': -1.0 }

        vw = SourceVecWrapper(unknowns, states)

        self.assertEqual(vw.vec.size, 8)
        self.assertEqual(len(vw), 4)
        self.assertEqual(vw.keys(), ['y1','y2','y3','s1'])
        print vw['y1']
        self.assertTrue(np.all(vw['y1']==np.ones((3,2))))


if __name__ == "__main__":
    unittest.main()

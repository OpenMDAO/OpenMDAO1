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

        vw = SourceVecWrapper(unknowns, states, initialize=True)

        self.assertEqual(vw.vec.size, 8)
        self.assertEqual(len(vw), 4)
        self.assertEqual(vw.keys(), ['y1','y2','y3','s1'])
        self.assertTrue(np.all(vw['y1']==np.ones((3,2))))
        self.assertTrue(vw['y2']==2.0)
        self.assertTrue(vw['y3']=='foo')
        self.assertTrue(vw['s1']==-1.0)

        vw['y1'] = np.ones((3,2))*3.
        vw['y2'] = 2.5
        vw['y3'] = 'bar'
        vw['s1'] = 5.

        self.assertTrue(np.all(vw['y1']==np.ones((3,2))*3.))
        self.assertTrue(vw['y2']==2.5)
        self.assertTrue(vw['y3']=='bar')
        self.assertTrue(vw['s1']==5.)

        # set with a different shaped array
        try:
            vw['y1'] = np.ones((3,3))
        except Exception as err:
            self.assertEqual(str(err),
                             "could not broadcast input array from shape (9) into shape (6)")
        else:
            self.fail("Exception expected")



if __name__ == "__main__":
    unittest.main()

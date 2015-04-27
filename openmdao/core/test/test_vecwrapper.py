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
        unknowns['y4'] = { 'shape': (2, 1) }
        states['s1'] = { 'val': -1.0 }

        vw = SourceVecWrapper(unknowns, states, initialize=True)

        self.assertEqual(vw.vec.size, 10)
        self.assertEqual(len(vw), 5)
        self.assertEqual(vw.keys(), ['y1','y2','y3', 'y4', 's1'])
        self.assertTrue(np.all(vw['y1']==np.ones((3,2))))
        self.assertEqual(vw['y2'], 2.0)
        self.assertEqual(vw['y3'], 'foo')
        self.assertTrue(np.all(vw['y4']==np.zeros((2,1))))
        self.assertEqual(vw['s1'], -1.0)

        vw['y1'] = np.ones((3,2))*3.
        vw['y2'] = 2.5
        vw['y3'] = 'bar'
        vw['y4'] = np.ones((2,1))*7.
        vw['s1'] = 5.

        self.assertTrue(np.all(vw['y1']==np.ones((3,2))*3.))
        self.assertTrue(np.all(vw['y4']==np.ones((2,1))*7.))
        self.assertEqual(vw['y2'], 2.5)
        self.assertEqual(vw['y3'], 'bar')
        self.assertEqual(vw['s1'], 5.)

        # set with a different shaped array
        try:
            vw['y1'] = np.ones((3,3))
        except Exception as err:
            self.assertEqual(str(err),
                             "could not broadcast input array from shape (9) into shape (6)")
        else:
            self.fail("Exception expected")

        params = OrderedDict()
        params['y1'] = { 'val': np.ones((3, 2)) }
        params['y2'] = { 'val': 2.0 }
        params['y3'] = { 'val': "foo" }
        params['y4'] = { 'shape': (2, 1) }

        tvw = TargetVecWrapper(params, vw)

        self.assertEqual(tvw.vec.size, 9)
        self.assertEqual(len(tvw), 4)
        self.assertEqual(tvw.keys(), ['y1','y2','y3', 'y4'])
        self.assertTrue(np.all(tvw['y1']==np.ones((3,2))*3.))
        self.assertEqual(tvw['y2'], 2.5)
        self.assertEqual(tvw['y3'], 'bar')
        self.assertTrue(np.all(tvw['y4']==np.ones((2,1))*7.))

if __name__ == "__main__":
    unittest.main()

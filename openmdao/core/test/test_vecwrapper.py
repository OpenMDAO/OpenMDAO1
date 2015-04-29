import unittest
import numpy as np
from collections import OrderedDict

from openmdao.core.vecwrapper import VecWrapper

class TestVecWrapper(unittest.TestCase):

    def test_vecwrapper(self):
        outputs = OrderedDict()
        states = OrderedDict()

        outputs['y1'] = { 'val': np.ones((3, 2)) }
        outputs['y2'] = { 'val': 2.0 }
        outputs['y3'] = { 'val': "foo" }
        outputs['y4'] = { 'shape': (2, 1) }
        states['s1'] = { 'val': -1.0 }

        u = VecWrapper.create_source_vector(outputs, states, store_noflats=True)

        self.assertEqual(u.vec.size, 10)
        self.assertEqual(len(u), 5)
        self.assertEqual(u.keys(), ['y1','y2','y3', 'y4', 's1'])
        self.assertTrue(np.all(u['y1']==np.ones((3,2))))
        self.assertEqual(u['y2'], 2.0)
        self.assertEqual(u['y3'], 'foo')
        self.assertTrue(np.all(u['y4']==np.zeros((2,1))))
        self.assertEqual(u['s1'], -1.0)

        u['y1'] = np.ones((3,2))*3.
        u['y2'] = 2.5
        u['y3'] = 'bar'
        u['y4'] = np.ones((2,1))*7.
        u['s1'] = 5.

        self.assertTrue(np.all(u['y1']==np.ones((3,2))*3.))
        self.assertTrue(np.all(u['y4']==np.ones((2,1))*7.))
        self.assertEqual(u['y2'], 2.5)
        self.assertEqual(u['y3'], 'bar')
        self.assertEqual(u['s1'], 5.)

        # set with a different shaped array
        try:
            u['y1'] = np.ones((3,3))
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

        p = VecWrapper.create_target_vector(params, u, store_noflats=True)

        self.assertEqual(p.vec.size, 9)
        self.assertEqual(len(p), 4)
        self.assertEqual(p.keys(), ['y1','y2','y3', 'y4'])
        self.assertTrue(np.all(p['y1']==np.zeros((3,2))))
        self.assertEqual(p['y2'], 0.)
        self.assertEqual(p['y3'], 'bar')
        self.assertTrue(np.all(p['y4']==np.zeros((2,1))))

        p['y1'] = np.ones((3,2))*9.
        self.assertTrue(np.all(p['y1']==np.ones((3,2))*9.))

    def test_view(self):
        # TODO: test VecWrapper.get_view()
        self.fail("Test not yet implemented")


if __name__ == "__main__":
    unittest.main()

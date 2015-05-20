import unittest
import numpy as np
from collections import OrderedDict

from openmdao.core.vecwrapper import SrcVecWrapper, TgtVecWrapper

class TestVecWrapper(unittest.TestCase):

    def test_vecwrapper(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'val': np.ones((3, 2)) }
        unknowns_dict['y2'] = { 'val': 2.0 }
        unknowns_dict['y3'] = { 'val': "foo" }
        unknowns_dict['y4'] = { 'shape': (2, 1), }
        unknowns_dict['s1'] = { 'val': -1.0, 'state': True, }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        self.assertEqual(u.vec.size, 10)
        self.assertEqual(len(u), 5)
        self.assertEqual(list(u.keys()), ['y1','y2','y3', 'y4', 's1'])
        self.assertTrue(np.all(u['y1']==np.ones((3,2))))
        self.assertEqual(u['y2'], 2.0)
        self.assertEqual(u['y3'], 'foo')
        self.assertTrue(np.all(u['y4']==np.zeros((2,1))))
        self.assertEqual(u['s1'], -1.0)

        self.assertEqual(u.get_states(), ['s1'])
        self.assertEqual([t[0] for t in u.get_vecvars()], ['y1','y2','y4','s1'])
        self.assertEqual([t[0] for t in u.get_byobjs()], ['y3'])

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

        for p, meta in params.items():
            meta['pathname'] = p
            meta['relative_name'] = p

        connections = {}
        for p in params:
            connections[p] = p

        p = TgtVecWrapper()
        p.setup(None, params, u, params.keys(),
                                            connections, store_byobjs=True)

        self.assertEqual(p.vec.size, 9)
        self.assertEqual(len(p), 4)
        self.assertEqual(list(p.keys()), ['y1','y2','y3', 'y4'])
        self.assertTrue(np.all(p['y1']==np.zeros((3,2))))
        self.assertEqual(p['y2'], 0.)
        self.assertEqual(p['y3'], 'bar')
        self.assertTrue(np.all(p['y4']==np.zeros((2,1))))

        p['y1'] = np.ones((3,2))*9.
        self.assertTrue(np.all(p['y1']==np.ones((3,2))*9.))

    def test_view(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['C1:y1'] = { 'val': np.ones((3, 2)) }
        unknowns_dict['C1:y2'] = { 'val': 2.0 }
        unknowns_dict['C1:y3'] = { 'val': "foo" }
        unknowns_dict['C2:y4'] = { 'shape': (2, 1), }
        unknowns_dict['C2:s1'] = { 'val': -1.0, 'state': True, }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        varmap = {
            'C1:y1':'y1',
            'C1:y2':'y2',
            'C1:y3':'y3',
        }

        uview = u.get_view(None, varmap)

        self.assertEqual(list(uview.keys()), ['y1', 'y2', 'y3'])

        uview['y2'] = 77.
        uview['y3'] = 'bar'

        self.assertEqual(uview['y2'], 77.)
        self.assertEqual(u['C1:y2'], 77.)

        self.assertEqual(uview['y3'], 'bar')
        self.assertEqual(u['C1:y3'], 'bar')

        # now get a view that's empty
        uview2 = u.get_view(None, {})
        self.assertEqual(list(uview2.keys()), [])

    def test_flat(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['C1:y1'] = { 'val': np.ones((3, 2)) }
        unknowns_dict['C1:y2'] = { 'val': 2.0 }
        unknowns_dict['C1:y3'] = { 'val': "foo" }
        unknowns_dict['C2:y4'] = { 'shape': (2, 1), }
        unknowns_dict['C2:s1'] = { 'val': -1.0, 'state': True, }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        self.assertTrue((np.array(u.flat['C1:y1'])==np.array([1., 1., 1., 1., 1., 1.])).all())
        self.assertTrue((np.array(u.flat['C1:y2'])==np.array([2.])).all())

        try:
            u.flat['C1:y3']
        except Exception as err:
            self.assertEqual(str(err), "'C1:y3' is a 'pass by object' variable. Flat value not found.")
        else:
            self.fail('Exception expected')
        self.assertTrue((np.array(u.flat['C2:y4'])==np.array([0., 0.])).all())
        self.assertTrue((np.array(u.flat['C2:s1'])==np.array([-1.])).all())

    def test_norm(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'val' : np.array([2.0, 3.0]) }
        unknowns_dict['y2'] = { 'val' : -4.0 }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        unorm = u.norm()
        self.assertAlmostEqual(unorm, np.linalg.norm(np.array([2.0, 3.0, -4.0])))

    def test_bad_get_unknown(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'val': np.ones((3, 2)) }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        try:
            u['A:y1']
        except KeyError as err:
            self.assertEqual(str(err), '"Variable \'A:y1\' does not exist"')
        else:
            self.fail('KeyError expected')

    def test_bad_set_unknown(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'val': np.ones((3, 2)) }

        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['relative_name'] = u

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        u = SrcVecWrapper()
        u.setup(unknowns_dict, store_byobjs=True)

        try:
            u['A:y1'] = np.zeros((3, 2))
        except KeyError as err:
            self.assertEqual(str(err), '"Variable \'A:y1\' does not exist"')
        else:
            self.fail('KeyError expected')

if __name__ == "__main__":
    unittest.main()

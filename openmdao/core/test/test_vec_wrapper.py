""" Tests for the OpenmDAO vecwrappers."""

import unittest
import numpy as np
from six import iteritems
from collections import OrderedDict

from openmdao.core.vec_wrapper import SrcVecWrapper, TgtVecWrapper
from openmdao.core.system import System, _SysData
from openmdao.core.problem import _ProbData

pbd = _ProbData()

class TestVecWrapper(unittest.TestCase):

    def test_vecwrapper(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }
        unknowns_dict['y2'] = { 'shape': 1, 'size': 1, 'val': 2.0 }
        unknowns_dict['y3'] = { 'size': 0, 'val': "foo", 'pass_by_obj': True }
        unknowns_dict['y4'] = { 'shape': (2,1), 'size': 2, 'val': np.zeros((2, 1)), }
        unknowns_dict['s1'] = { 'shape': 1, 'size': 1, 'val': -1.0, 'state': True, }

        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        self.assertEqual(u.vec.size, 10)
        self.assertEqual(len(u), 5)
        self.assertEqual(list(u.keys()), ['y1','y2','y3', 'y4', 's1'])
        self.assertTrue(np.all(u['y1']==np.ones((3,2))))
        self.assertEqual(u['y2'], 2.0)
        self.assertEqual(u['y3'], 'foo')
        self.assertTrue(np.all(u['y4']==np.zeros((2,1))))
        self.assertEqual(u['s1'], -1.0)

        self.assertEqual([t[0] for t in u.vec_val_iter()], ['y1','y2','y4','s1'])

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
        params['y1'] = { 'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }
        params['y2'] = { 'shape': 1, 'size': 1, 'val': 2.0 }
        params['y3'] = { 'size': 0, 'val': "foo" }
        params['y4'] = { 'shape': (2,1), 'size': 6, 'val': np.zeros((2, 1)) }

        for p, meta in params.items():
            meta['pathname'] = p
            meta['top_promoted_name'] = p
            sd.to_prom_name[u] = u

        connections = {}
        for p in params:
            connections[p] = (p, None)

        s = _SysData('')
        s._unknowns_dict = u._dat
        p = TgtVecWrapper(s, pbd)
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

        unknowns_dict['C1.y1'] = { 'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }
        unknowns_dict['C1.y2'] = { 'shape': 1, 'size': 1, 'val': 2.0 }
        unknowns_dict['C1.y3'] = { 'size': 0, 'val': "foo", 'pass_by_obj': True }
        unknowns_dict['C2.y4'] = { 'shape': (2, 1),  'val': np.zeros((2,1)), 'size': 2,  }
        unknowns_dict['C2.s1'] = { 'shape': 1, 'size': 1, 'val': -1.0, 'state': True, }


        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        varmap = OrderedDict([
            ('C1.y1','y1'),
            ('C1.y2','y2'),
            ('C1.y3','y3'),
        ])

        s = System()
        s._sysdata = _SysData('')
        s._probdata = pbd
        uview = u.get_view(s, None, varmap)

        self.assertEqual(list(uview.keys()), ['y1', 'y2', 'y3'])

        uview['y2'] = 77.
        uview['y3'] = 'bar'

        self.assertEqual(uview['y2'], 77.)
        self.assertEqual(u['C1.y2'], 77.)

        self.assertEqual(uview['y3'], 'bar')
        self.assertEqual(u['C1.y3'], 'bar')

        # now get a view that's empty
        uview2 = u.get_view(s, None, {})
        self.assertEqual(list(uview2.keys()), [])

    def test_flat(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['C1.y1'] = { 'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }
        unknowns_dict['C1.y2'] = { 'shape': 1, 'size': 1, 'val': 2.0 }
        unknowns_dict['C1.y3'] = { 'size': 0, 'val': "foo", 'pass_by_obj': True }
        unknowns_dict['C2.y4'] = { 'shape': (2,1), 'size': 2, 'val': np.zeros((2, 1)), }
        unknowns_dict['C2.s1'] = { 'shape': 1, 'size': 1, 'val': -1.0, 'state': True, }

        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        self.assertTrue((np.array(u._dat['C1.y1'].val)==np.array([1., 1., 1., 1., 1., 1.])).all())
        self.assertTrue((np.array(u._dat['C1.y2'].val)==np.array([2.])).all())

    def test_norm(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'shape': (2,1), 'size': 2, 'val' : np.array([2.0, 3.0]) }
        unknowns_dict['y2'] = { 'shape': 1, 'size': 1, 'val' : -4.0 }

        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        unorm = u.norm()
        self.assertAlmostEqual(unorm, np.linalg.norm(np.array([2.0, 3.0, -4.0])))

    def test_bad_get_unknown(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = { 'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }

        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        try:
            u['A.y1']
        except KeyError as err:
            self.assertEqual(str(err), "'A.y1'")
        else:
            self.fail('KeyError expected')

    def test_bad_set_unknown(self):
        unknowns_dict = OrderedDict()

        unknowns_dict['y1'] = {  'shape': (3,2), 'size': 6, 'val': np.ones((3, 2)) }

        sd = _SysData('')
        for u, meta in unknowns_dict.items():
            meta['pathname'] = u
            meta['top_promoted_name'] = u
            sd.to_prom_name[u] = u

        u = SrcVecWrapper(sd, pbd)
        u.setup(unknowns_dict, store_byobjs=True)

        try:
            u['A.y1'] = np.zeros((3, 2))
        except KeyError as err:
            self.assertEqual(str(err), "'A.y1'")
        else:
            self.fail('KeyError expected')

if __name__ == "__main__":
    unittest.main()

""" Test for the Component class"""

import unittest
from collections import OrderedDict
from six import text_type

import numpy as np

from openmdao.core.component import Component

class TestComponent(unittest.TestCase):

    def setUp(self):
        self.comp = Component()
        
    def test_properties(self):
        self.comp.add_param('x', 1)
        self.comp.add_state('y', 1)
        self.comp.add_output('z', 1)
        
        self.assertEqual(self.comp.params, OrderedDict([('x', {'val': 1})]))
        self.assertEqual(self.comp.unknowns, OrderedDict([('y', {'state': True, 'val': 1}), ('z', {'val': 1})]))
    def test_promotes(self):
        self.comp.add_param("xxyyzz", 0.0)
        self.comp.add_param("foobar", 0.0)
        self.comp.add_output("a.bcd.efg", -1)
        self.comp.add_output("x_y_z", np.zeros(10))

        self.comp._promotes = ('*',)
        for name in self.comp._params_dict:
            self.assertTrue(self.comp.promoted(name))
        for name in self.comp._unknowns_dict:
            self.assertTrue(self.comp.promoted(name))

        self.assertFalse(self.comp.promoted('blah'))

        self.comp._promotes = ('x*',)
        for name in self.comp._params_dict:
            if name.startswith('x'):
                self.assertTrue(self.comp.promoted(name))
            else:
                self.assertFalse(self.comp.promoted(name))
        for name in self.comp._unknowns_dict:
            if name.startswith('x'):
                self.assertTrue(self.comp.promoted(name))
            else:
                self.assertFalse(self.comp.promoted(name))

        self.comp._promotes = ('*.efg',)
        for name in self.comp._params_dict:
            if name.endswith('.efg'):
                self.assertTrue(self.comp.promoted(name))
            else:
                self.assertFalse(self.comp.promoted(name))
        for name in self.comp._unknowns_dict:
            if name.endswith('.efg'):
                self.assertTrue(self.comp.promoted(name))
            else:
                self.assertFalse(self.comp.promoted(name))

        # catch bad type on _promotes
        try:
            self.comp._promotes = ('*')
            self.comp.promoted('xxyyzz')
        except Exception as err:
            self.assertEqual(text_type(err),
                             " promotes must be specified as a list, tuple or other iterator of strings, but '*' was specified")

    def test_add_params(self):
        self.comp.add_param("x", 0.0)
        self.comp.add_param("y", 0.0)

        params, unknowns = self.comp._setup_variables()

        self.assertEquals(["x", "y"], list(params.keys()))

        self.assertEquals(params["x"], {"val": 0.0, 'relative_name': 'x' })
        self.assertEquals(params["y"], {"val": 0.0, 'relative_name': 'y' })

    def test_add_outputs(self):
        self.comp.add_output("x", -1)
        self.comp.add_output("y", np.zeros(10))

        params, unknowns = self.comp._setup_variables()

        self.assertEquals(["x", "y"], list(unknowns.keys()))

        self.assertIsInstance(unknowns["x"]["val"], int)
        self.assertIsInstance(unknowns["y"]["val"], np.ndarray)

        self.assertEquals(unknowns["x"], {"val": -1, 'relative_name': 'x' })
        self.assertEquals(list(unknowns["y"]["val"]), 10*[0])

    def test_add_states(self):
        self.comp.add_state("s1", 0.0)
        self.comp.add_state("s2", 6.0)

        params, unknowns = self.comp._setup_variables()

        self.assertEquals(["s1", "s2"], list(unknowns.keys()))

        self.assertEquals(unknowns["s1"], {"val": 0.0, 'state': True, 'relative_name': 's1' })
        self.assertEquals(unknowns["s2"], {"val": 6.0, 'state': True, 'relative_name': 's2' })

if __name__ == "__main__":
    unittest.main()

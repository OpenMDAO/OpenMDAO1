""" Test for the Component class"""

import unittest

import numpy as np

from openmdao.core.component import Component

class TestComponent(unittest.TestCase):

    def setUp(self):
        self.comp = Component()

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

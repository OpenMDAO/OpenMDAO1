import unittest
import numpy as np
from openmdao.core.component import Component

class TestComponent(unittest.TestCase):

    def setUp(self):
        self.comp = Component()

    def test_add_params(self):
        self.comp.add_param("x", 0.0)
        self.comp.add_param("y", 0.0)

        params, unknowns, states = self.comp.variables()

        self.assertEquals(["x", "y"], params.keys())

        self.assertEquals(params["x"], {"val" : 0.0})
        self.assertEquals(params["y"], {"val" : 0.0})

    def test_add_unknowns(self):
        self.comp.add_unknown("x", -1)
        self.comp.add_unknown("y", np.zeros(10))

        params, unknowns, states = self.comp.variables()

        self.assertEquals(["x", "y"], unknowns.keys())

        self.assertIsInstance(unknowns["x"]["val"], int)
        self.assertIsInstance(unknowns["y"]["val"], np.ndarray)

        self.assertEquals(unknowns["x"], {"val" : -1})
        self.assertEquals(list(unknowns["y"]["val"]), 10*[0])

    def test_add_states(self):
        self.comp.add_state("s1", 0.0)
        self.comp.add_state("s2", 6.0)

        params, unknowns, states = self.comp.variables()

        self.assertEquals(["s1", "s2"], states.keys())

        self.assertEquals(states["s1"], {"val" : 0.0})
        self.assertEquals(states["s2"], {"val" : 6.0})

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
from openmdao.core.system import System


class TestSystem(unittest.TestCase):
    

    def test_linearsystem(self):
        s = System()

        assert True

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
from openmdao.components.linear_system import LinearSystem
from openmdao.core.problem import Problem
from openmdao.core.group import Group

class TestProblem(unittest.TestCase):

    def test_hanging_params(self):

        root  = Group()
        root.add('ls', LinearSystem(size=10))

        prob = Problem(root)

        try:
            prob.setup()
        except Exception as error:
            self.assertEquals(error.message, 
                "Parameters ['ls:A','ls:b'] have no associated unknowns.")
        else:
            self.fail("Error expected")

if __name__ == "__main__":
    unittest.main()
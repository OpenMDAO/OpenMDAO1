""" Unit test for the Problem class. """

import unittest

import numpy as np

from openmdao.components.linear_system import LinearSystem
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.test.simplecomps import SimpleComp


class TestProblem(unittest.TestCase):

    def test_hanging_params(self):

        root  = Group()
        root.add('ls', LinearSystem(size=10))

        prob = Problem(root=root)

        try:
            prob.setup()
        except Exception as error:
            self.assertEquals(error.message,
                "Parameters ['ls:A','ls:b'] have no associated unknowns.")
        else:
            self.fail("Error expected")

    def test_calc_gradient_interface_errors(self):

        root  = Group()
        prob = Problem(root=root)
        root.add('comp', SimpleComp())

        try:
            prob.calc_gradient(['comp:x'], ['comp:y'], mode='junk')
        except Exception as error:
            msg = "mode must be 'auto', 'fwd', or 'rev'"
            self.assertEquals(error.message, msg)
        else:
            self.fail("Error expected")

        try:
            prob.calc_gradient(['comp:x'], ['comp:y'], return_format='junk')
        except Exception as error:
            msg = "return_format must be 'array' or 'dict'"
            self.assertEquals(error.message, msg)
        else:
            self.fail("Error expected")

    def test_setup(self):
        self.fail("Not Implemented yet")

if __name__ == "__main__":
    unittest.main()
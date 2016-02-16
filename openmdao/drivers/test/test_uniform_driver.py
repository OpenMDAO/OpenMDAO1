"""Testing UniformDriver"""

import unittest
from pprint import pformat
from types import GeneratorType

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem
from openmdao.test.paraboloid import Paraboloid

from openmdao.drivers.uniform_driver import UniformDriver

class TestUniformDriver(unittest.TestCase):

    def test_uniformDriver(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = UniformDriver(5)
        prob.driver.add_desvar('x', lower=-10, upper=10)
        prob.driver.add_desvar('y', lower=-10, upper=10)
        prob.driver.add_objective('f_xy')

        prob.setup(check=False)
        runList = prob.driver._build_runlist()
        prob.run()

        # Assert that the runList generated is of Generator Type
        self.assertTrue((type(runList) == GeneratorType),
                        "_build_runlist did not return a generator.")

        # Assert the length is correct
        cases =list(runList)
        countRuns = len(cases)
        self.assertTrue(countRuns == 5, "Incorrect number of runs generated.")

        #assert that values are in range
        inRange= True
        for value in cases:
            value = dict(value)
            if not (-10< value['x'] < 10 and -10 < value['y'] <10):
                inRange = False

        self.assertTrue(inRange,"Not in range.")

if __name__ == "__main__":
    unittest.main()

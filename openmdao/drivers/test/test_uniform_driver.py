import unittest
from pprint import pformat
from random import randint
from types import GeneratorType

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivatives, SellarStateConnection
from openmdao.test.simple_comps import SimpleArrayComp, ArrayComp2D
from openmdao.test.util import assert_rel_error

from openmdao.drivers.uniform_driver import UniformDriver

class TestFullFactorial(unittest.TestCase):
    def test_fullfactorial(self):
        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])

        prob.driver = UniformDriver(5)
        prob.driver.add_desvar('x', low=-10, high=10)
        prob.driver.add_desvar('y', low=-10, high=10)
        prob.driver.add_objective('f_xy')
        runList = prob.driver._build_runlist()
        prob.setup(check=False)
        prob.run()
        # Assert that the runList generated is of Generator Type
        self.assertTrue((type(runList) == GeneratorType),"_build_runlist did not return a generator.")

        # Assert the length is correct
        cases =list(runList)
        countRuns = len(cases)
        self.assertTrue(countRuns == 5, "Incorrect number of runs generated.")

        #assert that values are in range
        inRange= False
        for value in cases:
            if (-10< value['x'] < 10 and -10 <value['y'] <10):
                inRange = True

        self.assertTrue(inRange,"Not in range.")

if __name__ == "__main__":
    unittest.main()


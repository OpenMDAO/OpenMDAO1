""" Testing pyoptsparse SNOPT."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simplecomps import SimpleArrayComp
from openmdao.test.testutil import assert_rel_error

SKIP = False
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    # Just so python can parse this file.
    from openmdao.core.driver import Driver
    pyOptSparseDriver = Driver
    SKIP = True


class TestPyoptSparse(unittest.TestCase):

    def setUp(self):
        if SKIP is True:
            raise unittest.SkipTest

    def test_simple_paraboloid(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        top.driver = pyOptSparseDriver()
        top.driver.add_param('x', low=-50.0, high=50.0)
        top.driver.add_param('y', low=-50.0, high=50.0)

        top.driver.add_objective('f_xy')
        top.driver.add_constraint('c')

        top.setup()
        top.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, top['x'], 7.16667, 1e-6)
        assert_rel_error(self, top['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', 50.0), promotes=['*'])
        root.add('p2', ParamComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = 15.0 - x + y'), promotes=['*'])

        top.driver = pyOptSparseDriver()
        top.driver.add_param('x', low=-50.0, high=50.0)
        top.driver.add_param('y', low=-50.0, high=50.0)

        top.driver.add_objective('f_xy')
        top.driver.add_constraint('c', ctype='ineq')

        top.setup()
        top.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, top['x'], 7.16667, 1e-6)
        assert_rel_error(self, top['y'], -7.833334, 1e-6)

    def test_simple_array_comp(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', np.zeros([2])), promotes=['*'])
        root.add('comp', SimpleArrayComp(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.array([0.0, 0.0]), y=np.array([0.0, 0.0])), promotes=['*'])
        root.add('obj', ExecComp('o = y[0]', y=np.array([0.0, 0.0])), promotes=['*'])

        top.driver = pyOptSparseDriver()
        top.driver.add_param('x', low=-50.0, high=50.0)

        top.driver.add_objective('o')
        top.driver.add_constraint('c', ctype='eq')

        top.setup()
        top.run()

        obj = top['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

if __name__ == "__main__":
    unittest.main()
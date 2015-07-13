""" Testing pyoptsparse SNOPT."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simplecomps import SimpleArrayComp, ArrayComp2D
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
            raise unittest.SkipTest("Could not import pyOptSparseDriver. "
                                    "Is pyoptsparse installed?")

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

    def test_simple_array_comp2D(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', np.zeros((2, 2))), promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.zeros((2, 2)), y=np.zeros((2, 2))), promotes=['*'])
        root.add('obj', ExecComp('o = y[0, 0]', y=np.zeros((2, 2))), promotes=['*'])

        top.driver = pyOptSparseDriver()
        top.driver.add_param('x', low=-50.0, high=50.0)

        top.driver.add_objective('o')
        top.driver.add_constraint('c', ctype='eq')

        top.setup()
        top.run()

        obj = top['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', np.zeros((2, 2))), promotes=['*'])
        root.add('comp', ArrayComp2D(), promotes=['*'])
        root.add('con', ExecComp('c = y - 20.0', c=np.zeros((2, 2)), y=np.zeros((2, 2))), promotes=['*'])
        root.add('obj', ExecComp('o = y[0, 0]', y=np.zeros((2, 2))), promotes=['*'])

        top.driver = pyOptSparseDriver()
        top.driver.add_param('x', low=-50.0*np.ones((2, 2)), high=50.0*np.ones((2, 2)))

        top.driver.add_objective('o')
        top.driver.add_constraint('c', ctype='eq')

        top.setup()
        top.run()

        obj = top['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_fan_out(self):

        top = Problem()
        root = top.root = Group()

        root.add('p1', ParamComp('x', 1.0))
        root.add('p2', ParamComp('x', 1.0))

        root.add('comp1', ExecComp('y = 3.0*x'))
        root.add('comp2', ExecComp('y = 5.0*x'))

        root.add('obj', ExecComp('o = i1 + i2'))
        root.add('con1', ExecComp('c = 15.0 - x'))
        root.add('con2', ExecComp('c = 15.0 - x'))

        # hook up non explicitly
        root.connect('p1.x', 'comp1.x')
        root.connect('p2.x', 'comp2.x')
        root.connect('comp1.y', 'obj.i1')
        root.connect('comp2.y', 'obj.i2')
        root.connect('comp1.y', 'con1.x')
        root.connect('comp2.y', 'con2.x')

        top.driver = pyOptSparseDriver()
        top.driver.add_param('p1.x', low=-50.0, high=50.0)
        top.driver.add_param('p2.x', low=-50.0, high=50.0)
        top.driver.add_objective('obj.o')
        top.driver.add_constraint('con1.c', ctype='eq')
        top.driver.add_constraint('con2.c', ctype='eq')

        top.setup()
        top.run()

        obj = top['obj.o']
        assert_rel_error(self, obj, 30.0, 1e-6)

        # Verify that pyOpt has the correct wrt names
        con1 = top.driver.pyopt_solution.constraints['con1.c']
        self.assertEqual(con1.wrt, ['p1.x'])
        con2 = top.driver.pyopt_solution.constraints['con2.c']
        self.assertEqual(con2.wrt, ['p2.x'])

if __name__ == "__main__":
    unittest.main()

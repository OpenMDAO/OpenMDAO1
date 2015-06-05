""" Unit test for the Problem class. """

import unittest
import numpy as np
from six import text_type, PY3
import warnings

from openmdao.components.linear_system import LinearSystem
from openmdao.core.component import Component
from openmdao.core.problem import Problem
from openmdao.core.checks import ConnectError
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.test.examplegroups import ExampleGroup, ExampleGroupWithPromotes, ExampleByObjGroup

if PY3:
    def py3fix(s):
        return s.replace('<type', '<class')
else:
    def py3fix(s):
        return s

class TestProblem(unittest.TestCase):

    def test_conflicting_connections(self):
        # verify we get an error if we have conflicting implicit and explicit connections
        root = Group()

        # promoting G1:x will create an implicit connection to G3:x
        # this is a conflict because G3:x (aka G3:C4:x) is already connected
        # to G3:C3:x
        G2 = root.add('G2', Group(), promotes=['x'])  # BAD PROMOTE
        G2.add('C1', ParamComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', ExecComp('y=x*2.0'), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', ExecComp('y=x*2.0'))
        G3.add('C4', ExecComp('y=x*2.0'), promotes=['x'])

        root.connect('G2:G1:C2:y', 'G3:C3:x')
        G3.connect('C3:y', 'x')

        prob = Problem(root)

        try:
            prob.setup()
        except Exception as error:
            msg = "'G3:C4:x' is explicitly connected to 'G3:C3:y' but implicitly connected to 'G2:C1:x'"
            self.assertEquals(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_conflicting_promotions(self):
        # verify we get an error if we have conflicting promotions
        root = Group()

        # promoting G1:x will create an implicit connection to G3:x
        # this is a conflict because G3:x (aka G3:C4:x) is already connected
        # to G3:C3:x
        G2 = root.add('G2', Group())
        G2.add('C1', ParamComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', ExecComp('y=x*2.0'), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', ExecComp('y=x*2.0'), promotes=['y'])          # promoting y
        G3.add('C4', ExecComp('y=x*2.0'), promotes=['x', 'y'])     # promoting y again.. BAD

        prob = Problem(root)

        try:
            prob.setup()
        except Exception as error:
            msg = "Promoted name 'G3:y' matches multiple unknowns: ['G3:C3:y', 'G3:C4:y']"
            self.assertEquals(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_hanging_params(self):
        # test that a warning is issued for an unconnected parameter
        root  = Group()
        root.add('ls', LinearSystem(size=10))

        prob = Problem(root=root)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.setup()
            assert len(w) == 1, "Warning expected."
            self.assertEquals("Parameters ['ls:A', 'ls:b'] have no associated unknowns.",
                              str(w[-1].message))

    def test_unconnected_param_access(self):
        prob = Problem(root=Group())
        G1 = prob.root.add("G1", Group())
        G2 = G1.add("G2", Group())
        C1 = G2.add("C1", ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        C2 = G2.add("C2", ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        G2.connect("C1:y", "C2:x")

        prob.setup()
        prob.run()

        C1.params['x'] = 2.
        self.assertEqual(prob['G1:G2:C1:x'], 2.0)
        prob['G1:G2:C1:x'] = 99.
        self.assertEqual(C1.params['x'], 99.)

    def test_unconnected_param_access_with_promotes(self):
        prob = Problem(root=Group())
        G1 = prob.root.add("G1", Group())
        G2 = G1.add("G2", Group(), promotes=['x'])
        C1 = G2.add("C1", ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']), promotes=['x'])
        C2 = G2.add("C2", ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        G2.connect("C1:y", "C2:x")

        prob.setup()
        prob.run()

        # still must use absolute naming to find params even if they're
        # promoted.  Promoted names for params can refer to more than one param.
        C1.params['x'] = 2.
        self.assertEqual(prob['G1:G2:C1:x'], 2.0)
        self.assertEqual(prob.root['G1:G2:C1:x'], 2.0)
        prob['G1:G2:C1:x'] = 99.
        self.assertEqual(C1.params['x'], 99.)
        prob.root.subsystem('G1')['G2:C1:x'] = 12.
        self.assertEqual(C1.params['x'], 12.)

        try:
            prob['G1:x'] = 11.
        except Exception as err:
            self.assertEqual(err.args[0],
                             "Can't find variable 'G1:x' in unknowns or params vectors in system ''")
        else:
            self.fail("exception expected")

    def test_calc_gradient_interface_errors(self):

        root  = Group()
        prob = Problem(root=root)
        root.add('comp', ExecComp('y=x*2.0'))

        try:
            prob.calc_gradient(['comp:x'], ['comp:y'], mode='junk')
        except Exception as error:
            msg = "mode must be 'auto', 'fwd', 'rev', or 'fd'"
            self.assertEquals(text_type(error), msg)
        else:
            self.fail("Error expected")

        try:
            prob.calc_gradient(['comp:x'], ['comp:y'], return_format='junk')
        except Exception as error:
            msg = "return_format must be 'array' or 'dict'"
            self.assertEquals(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_check_connections(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_state('y', np.zeros((2,)), shape=(2,))

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('y', np.zeros((3,)), shape=(3,))

        class C(Component):
            def __init__(self):
                super(C, self).__init__()
                self.add_state('y', np.zeros((2,)))

        class D(Component):
            def __init__(self):
                super(D, self).__init__()
                self.add_param('y', np.zeros((2,)))

        class E(Component):
            def __init__(self):
                super(E, self).__init__()
                self.add_param('y', 1.0)

        #Explicit
        expected_error_message = py3fix("Type '<type 'numpy.ndarray'>' of source "
                                  "'A:y' must be the same as type "
                                  "'<type 'float'>' of target "
                                  "'E:y'")
        problem = Problem()
        problem.root = Group()
        problem.root.add('A', A())
        problem.root.add('E', E())

        problem.root.connect('A:y', 'E:y')

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        #Implicit
        expected_error_message = py3fix("Type '<type 'numpy.ndarray'>' of source "
                                  "'y' must be the same as type "
                                  "'<type 'float'>' of target "
                                  "'y'")

        problem = Problem()
        problem.root = Group()
        problem.root.add('A', A(), promotes=['y'])
        problem.root.add('E', E(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)


        # Explicit
        expected_error_message = ("Shape '(2,)' of the source 'A:y' "
                                  "must match the shape '(3,)' "
                                  "of the target 'B:y'")
        problem = Problem()
        problem.root = Group()

        problem.root.add('A', A())
        problem.root.add('B', B())
        problem.root.connect('A:y', 'B:y')

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        # Implicit
        expected_error_message = ("Shape '(2,)' of the source 'y' "
                                  "must match the shape '(3,)' "
                                  "of the target 'y'")

        problem = Problem()
        problem.root = Group()

        problem.root.add('A', A(), promotes=['y'])
        problem.root.add('B', B(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        # Explicit
        expected_error_message = ("Shape '(2,)' of the source 'C:y' must match the shape '(3,)' "
                                  "of the target 'B:y'")

        problem = Problem()
        problem.root = Group()
        problem.root.add('B', B())
        problem.root.add('C', C())
        problem.root.connect('C:y', 'B:y')

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        # Implicit
        expected_error_message = ("Shape '(2,)' of the source 'y' must match the shape"
                                  " '(3,)' of the target 'y'")

        problem = Problem()
        problem.root = Group()
        problem.root.add('B', B(), promotes=['y'])
        problem.root.add('C', C(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        # Explicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('A', A())
        problem.root.add('D', D())
        problem.root.connect('A:y', 'D:y')
        problem.setup()

        # Implicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('A', A(), promotes=['y'])
        problem.root.add('D', D(), promotes=['y'])
        problem.setup()

        # Explicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('C', C())
        problem.root.add('D', D())
        problem.root.connect('C:y', 'D:y')
        problem.setup()

        # Implicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('C', C(), promotes=['y'])
        problem.root.add('D', D(), promotes=['y'])
        problem.setup()

    def test_simplest_run(self):

        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', ParamComp('x', 7.0))
        root.add('mycomp', ExecComp('y=x*2.0'))

        root.connect('x_param:x', 'mycomp:x')

        prob.setup()
        prob.run()
        result = root.unknowns['mycomp:y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_simplest_run_w_promote(self):

        prob = Problem(root=Group())
        root = prob.root

        # ? Didn't we say that ParamComp by default promoted its variable?
        root.add('x_param', ParamComp('x', 7.0), promotes=['x'])
        root.add('mycomp', ExecComp('y=x*2.0'), promotes=['x'])

        prob.setup()
        prob.run()
        result = root.unknowns['mycomp:y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_variable_access(self):
        prob = Problem(root=ExampleGroup())

        # set with a different shaped array
        try:
            prob['G2:C1:x']
        except Exception as err:
            msg = 'setup() must be called before variables can be accessed'
            self.assertEquals(text_type(err), msg)
        else:
            self.fail('Exception expected')

        prob.setup()

        self.assertEqual(prob['G2:C1:x'], 5.)                # default output from ParamComp
        self.assertEqual(prob['G2:G1:C2:y'], 5.5)            # output from ExecComp
        self.assertEqual(prob.root.subsystem('G3:C3').params['x'], 0.)      # initial value for a parameter
        self.assertEqual(prob.root.subsystem('G2:G1:C2').params['x'], 0.)   # initial value for a parameter

        prob = Problem(root=ExampleGroupWithPromotes())
        prob.setup()
        self.assertEqual(prob.root.subsystem('G2:G1:C2').params['x'], 0.)      # initial value for a parameter

        # __setitem__
        prob['G2:G1:C2:y'] = 99.
        self.assertEqual(prob['G2:G1:C2:y'], 99.)


    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        prob.setup()
        prob.run()

        self.assertAlmostEqual(prob['G3:C4:y'], 40.)

    def test_byobj_run(self):
        prob = Problem(root=ExampleByObjGroup())

        prob.setup()
        prob.run()

        self.assertEqual(prob['G3:C4:y'], 'fooC2C3C4')

    def test_scalar_sizes(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_param('x', shape=1)
                self.add_output('y', shape=1)

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('x', shape=2)
                self.add_output('y', shape=2)

        class C(Component):
            def __init__(self):
                super(C, self).__init__()
                self.add_param('x', shape=3)
                self.add_output('y', shape=3)

        # Scalar Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', 0., shape=1), promotes=['x'])
        root.add('A1', A(), promotes=['x'])
        root.add('A2', A())
        root.connect('A1:y', 'A2:x')
        prob.setup()

        # Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('B2', B())
        root.connect('B1:y', 'B2:x')
        prob.setup()

        # Mismatched Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('C1', C())
        root.connect('B1:y', 'C1:x')
        with self.assertRaises(ConnectError) as cm:
            prob.setup()
        expected_error_message = "Shape '(2,)' of the source "\
                                  "'B1:y' must match the shape '(3,)' "\
                                  "of the target 'C1:x'"
        self.assertEquals(expected_error_message, str(cm.exception))

        # Mismatched Scalar to Array Value
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', 0., shape=1), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        with self.assertRaises(ConnectError) as cm:
            prob.setup()

        expected_error_message = py3fix("Type '<type 'float'>' of source "
                                  "'x' must be the same as type "
                                  "'<type 'numpy.ndarray'>' of target "
                                  "'x'")
        self.assertEquals(expected_error_message, str(cm.exception))

if __name__ == "__main__":
    unittest.main()

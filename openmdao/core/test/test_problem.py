""" Unit test for the Problem class. """

import unittest
import numpy as np
from six import text_type, PY3

from openmdao.components.linear_system import LinearSystem
from openmdao.core.component import Component
from openmdao.core.problem import Problem
from openmdao.core.checks import ConnectError
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp
from openmdao.test.examplegroups import ExampleGroup, ExampleGroupWithPromotes, ExampleNoflatGroup
from openmdao.units.units import PhysicalQuantity

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
        G1.add('C2', SimpleComp(), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', SimpleComp())
        G3.add('C4', SimpleComp(), promotes=['x'])

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
        G1.add('C2', SimpleComp(), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', SimpleComp(), promotes=['y'])          # promoting y
        G3.add('C4', SimpleComp(), promotes=['x', 'y'])     # promoting y again.. BAD

        prob = Problem(root)

        try:
            prob.setup()
        except Exception as error:
            msg = "Promoted name 'G3:y' matches multiple unknowns: ['G3:C3:y', 'G3:C4:y']"
            self.assertEquals(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_hanging_params(self):

        root  = Group()
        root.add('ls', LinearSystem(size=10))

        prob = Problem(root=root)

        try:
            prob.setup()
        except Exception as error:
            self.assertEquals(text_type(error),
                "Parameters ['ls:A', 'ls:b'] have no associated unknowns.")
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
                self.add_param('y', np.zeros((3,)))

        class E(Component):
            def __init__(self):
                super(E, self).__init__()
                self.add_param('y', 1.0)

        class F(Component):
            def __init__(self):
                super(F, self).__init__()
                self.add_state('y', PhysicalQuantity("1m"))

        class G(Component):
            def __init__(self):
                super(G, self).__init__()
                self.add_param('y', PhysicalQuantity("1ft"))

        class H(Component):
            def __init__(self):
                super(H, self).__init__()
                self.add_param('y', PhysicalQuantity("1s"))

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
        expected_error_message = ("Shape of the initial value '(2,)' of source "
                                  "'C:y' must match the shape '(3,)' "
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
        expected_error_message = ("Shape of the initial value '(2,)' of source "
                                  "'y' must match the shape '(3,)' "
                                  "of the target 'y'")

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

        # Explicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('F', F())
        problem.root.add('G', G())
        problem.root.connect('F:y', 'G:y')
        problem.setup()

        # Implicit
        problem = Problem()
        problem.root = Group()
        problem.root.add('F', F(), promotes=['y'])
        problem.root.add('G', G(), promotes=['y'])
        problem.setup()

        # Explicit
        expected_error_message = py3fix("Unit '<PhysicalUnit m>' of the source "
                                  "'F:y' must be compatible with the unit "
                                  "'<PhysicalUnit s>' of the target "
                                  "'H:y'")
        problem = Problem()
        problem.root = Group()
        problem.root.add('F', F())
        problem.root.add('H', H())
        problem.root.connect('F:y', 'H:y')
        with self.assertRaises(ConnectError) as cm:
            problem.setup()

        self.assertEqual(str(cm.exception), expected_error_message)

        # Implicit
        expected_error_message = py3fix("Unit '<PhysicalUnit m>' of the source "
                                  "'y' must be compatible with the unit "
                                  "'<PhysicalUnit s>' of the target "
                                  "'y'")
        problem = Problem()
        problem.root = Group()
        problem.root.add('F', F(), promotes=['y'])
        problem.root.add('H', H(), promotes=['y'])
        with self.assertRaises(ConnectError) as cm:
            problem.setup()
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_simplest_run(self):

        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', ParamComp('x', 7.0))
        root.add('mycomp', SimpleComp())

        root.connect('x_param:x', 'mycomp:x')

        prob.setup()
        prob.run()
        result = root._varmanager.unknowns['mycomp:y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_simplest_run_w_promote(self):

        prob = Problem(root=Group())
        root = prob.root

        # ? Didn't we say that ParamComp by default promoted its variable?
        root.add('x_param', ParamComp('x', 7.0), promotes=['x'])
        root.add('mycomp', SimpleComp(), promotes=['x'])

        prob.setup()
        prob.run()
        result = root._varmanager.unknowns['mycomp:y']
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
        self.assertEqual(prob['G2:G1:C2:y'], 5.5)            # default output from SimpleComp
        self.assertEqual(prob['G3:C3:x', 'params'], 0.)      # initial value for a parameter
        self.assertEqual(prob['G2:G1:C2:x', 'params'], 0.)   # initial value for a parameter

        prob = Problem(root=ExampleGroupWithPromotes())
        prob.setup()
        self.assertEqual(prob['G2:G1:C2:x', 'params'], 0.)      # initial value for a parameter

        # __setitem__
        prob['G2:G1:C2:y'] = 99.
        self.assertEqual(prob['G2:G1:C2:y'], 99.)


    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        prob.setup()
        prob.run()

        self.assertAlmostEqual(prob['G3:C4:y'], 40.)

    def test_noflat_run(self):
        prob = Problem(root=ExampleNoflatGroup())

        prob.setup()
        prob.run()

        self.assertEqual(prob['G3:C4:y'], 'fooC2C3C4')



if __name__ == "__main__":
    unittest.main()

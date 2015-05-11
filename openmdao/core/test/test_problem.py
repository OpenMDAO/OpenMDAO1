""" Unit test for the Problem class. """

import unittest
import numpy as np
from six import text_type

from openmdao.components.linear_system import LinearSystem
from openmdao.core.component import Component
from openmdao.core.problem import ConnectError, Problem
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp


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
            msg = 'G3:C4:x is explicitly connected to G3:C3:y but implicitly connected to G2:C1:x'
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
            msg = "Promoted name G3:y matches multiple unknowns: ['G3:C3:y', 'G3:C4:y']"
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

        problem = Problem()
        problem.root = Group()

        problem.root.add('A', A())
        problem.root.add('B', B())
        problem.root.connect('A:y', 'B:y')

        expected_error_message = ("Shape '(2,)' of the source 'A:y' "
                                  "must match the shape '(3,)' "
                                  "of the target 'B:y'")

        with self.assertRaises(ConnectError) as cm:
            problem.setup()
        
        self.assertEqual(str(cm.exception), expected_error_message)
        
        problem = Problem()
        problem.root = Group()
        problem.root.add('B', B())
        problem.root.add('C', C())
        problem.root.connect('C:y', 'B:y')

        expected_error_message = ("Shape of the initial value '(2,)' of source "
                                  "'C:y' must match the shape '(3,)' "
                                  "of the target 'B:y'")
        
        with self.assertRaises(ConnectError) as cm:
            problem.setup()
        
        self.assertEqual(str(cm.exception), expected_error_message)

        problem = Problem()
        problem.root = Group()
        problem.root.add('A', A())
        problem.root.add('D', D())
        problem.root.connect('A:y', 'D:y')
        problem.setup()

        problem = Problem()
        problem.root = Group()
        problem.root.add('C', A())
        problem.root.add('D', D())
        problem.root.connect('C:y', 'D:y')
        problem.setup()
    #def test_basic_run(self):
        #prob = Problem(root=Group())
        #root = prob.root

        #G2 = root.add('G2', Group())
        #G2.add('C1', ParamComp('y1', 5.))

        #G1 = G2.add('G1', Group())
        #G1.add('C2', SimpleComp())

        #G3 = root.add('G3', Group())
        #G3.add('C3', SimpleComp())
        #G3.add('C4', SimpleComp())

        #G2.connect('C1:y1', 'G1:C2:x')
        #root.connect('G2:G1:C2:y', 'G3:C3:x')
        #G3.connect('C3:y', 'C4:x')

        #prob.setup()

        ## TODO: this needs Systems to be able to solve themselves

        # ...

if __name__ == "__main__":
    unittest.main()

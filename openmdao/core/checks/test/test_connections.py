import unittest
import numpy as np
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.checks.connections import ConnectError

class TestConnections(unittest.TestCase):
    def test_check_shapes_match(self):
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

if __name__ == "__main__":
    unittest.main()

""" Unit test for the Problem class. """

import sys
import unittest
import warnings

from six import text_type, PY3
from six.moves import cStringIO

import numpy as np

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         LinearGaussSeidel, ScipyGMRES, Driver
from openmdao.core.mpi_wrap import MPI
from openmdao.test.example_groups import ExampleGroup, ExampleGroupWithPromotes, ExampleByObjGroup
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.simple_comps import SimpleComp, SimpleImplicitComp, RosenSuzuki, FanIn
from openmdao.util.options import OptionsDictionary

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

        # promoting G1.x will create an implicit connection to G3.x
        # this is a conflict because G3.x (aka G3.C4.x) is already connected
        # to G3.C3.x
        G2 = root.add('G2', Group(), promotes=['x'])  # BAD PROMOTE
        G2.add('C1', IndepVarComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', ExecComp('y=x*2.0'), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', ExecComp('y=x*2.0'))
        G3.add('C4', ExecComp('y=x*2.0'), promotes=['x'])

        root.connect('G2.G1.C2.y', 'G3.C3.x')
        G3.connect('C3.y', 'x')

        prob = Problem(root)

        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "Target 'G3.C4.x' is connected to multiple unknowns: ['G2.C1.x', 'G3.C3.y']"
            self.assertTrue(msg in str(error))
        else:
            self.fail("Error expected")

    def test_check_promotes(self):
        # verify we get an error at setup time if we have promoted a var that doesn't exist

        # valid case, no error
        prob = Problem(Group())
        G = prob.root.add('G', Group())
        C = G.add('C', SimpleComp(), promotes=['x*', 'y'])
        # ignore warning about the unconnected param
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            prob.setup(check=False)

        # promoting a non-existent variable should throw an error
        prob = Problem(Group())
        G = prob.root.add('G', Group())
        C = G.add('C', SimpleComp(), promotes=['spoon'])        # there is no spoon
        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "'G.C' promotes 'spoon' but has no variables matching that specification"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

        # promoting a pattern with no matches should throw an error
        prob = Problem(Group())
        G = prob.root.add('G', Group())
        P = G.add('P', IndepVarComp('x', 5.), promotes=['a*'])     # there is no match
        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "'G.P' promotes 'a*' but has no variables matching that specification"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_conflicting_promotions(self):
        # verify we get an error if we have conflicting promotions
        root = Group()

        # promoting G1.x will create an implicit connection to G3.x
        # this is a conflict because G3.x (aka G3.C4.x) is already connected
        # to G3.C3.x
        G2 = root.add('G2', Group())
        G2.add('C1', IndepVarComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', ExecComp('y=x*2.0'), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', ExecComp('y=x*2.0'), promotes=['y'])          # promoting y
        G3.add('C4', ExecComp('y=x*2.0'), promotes=['x', 'y'])     # promoting y again.. BAD

        prob = Problem(root)

        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "'G3': promoted name 'y' matches multiple unknowns: ('G3.C3.y', 'G3.C4.y')"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_conflicting_promoted_state_vars(self):
        # verify we get an error if we have conflicting promoted state variables
        root = Group()

        comp1 = SimpleImplicitComp()
        comp2 = SimpleImplicitComp()

        root.add('c1', comp1, promotes=['z'])  # promote the state, z
        root.add('c2', comp2, promotes=['z'])  # promote the state, z, again.. BAD

        prob = Problem(root)

        with self.assertRaises(RuntimeError) as err:
            prob.setup(check=False)

        expected_msg = "'': promoted name 'z' matches multiple unknowns: ('c1.z', 'c2.z')"

        self.assertEqual(str(err.exception), expected_msg)

    def test_unconnected_param_access(self):
        prob = Problem(root=Group())
        G1 = prob.root.add('G1', Group())
        G2 = G1.add('G2', Group())
        C1 = G2.add('C1', ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        C2 = G2.add('C2', ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        G2.connect('C1.y', 'C2.x')

        # ignore warning about the unconnected param
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            prob.setup(check=False)

        prob.run()

        C1.params['x'] = 2.
        self.assertEqual(prob['G1.G2.C1.x'], 2.0)
        prob['G1.G2.C1.x'] = 99.
        self.assertEqual(C1.params['x'], 99.)

    def test_unconnected_param_access_with_promotes(self):
        prob = Problem(root=Group())
        G1 = prob.root.add('G1', Group())
        G2 = G1.add('G2', Group(), promotes=['x'])
        C1 = G2.add('C1', ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']), promotes=['x'])
        C2 = G2.add('C2', ExecComp(['y=2.0*x',
                                    'z=x*x-2.0']))
        G2.connect('C1.y', 'C2.x')

        # ignore warning about the unconnected param
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            prob.setup(check=False)

        prob.run()

        # still must use absolute naming to find params even if they're
        # promoted.  Promoted names for params can refer to more than one param.
        C1.params['x'] = 2.
        self.assertEqual(prob['G1.x'], 2.0)
        self.assertEqual(prob.root.G1.G2.C1.params['x'], 2.0)
        prob['G1.x'] = 99.
        self.assertEqual(C1.params['x'], 99.)
        prob['G1.x'] = 12.
        self.assertEqual(C1.params['x'], 12.)

        prob['G1.x'] = 17.

        self.assertEqual(prob.root.G1.G2.C1.params['x'], 17.0)
        prob.run()

    def test_input_input_explicit_conns_no_conn(self):
        prob = Problem(root=Group())
        root = prob.root
        root.add('p1', IndepVarComp('x', 1.0))
        root.add('c1', ExecComp('y = x*2.0'))
        root.add('c2', ExecComp('y = x*3.0'))
        root.connect('c1.x', 'c2.x')

        # ignore warning about the unconnected params
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            prob.setup(check=False)

        prob.run()
        self.assertEqual(root.connections, {})

    def test_input_input_explicit_conns_w_conn(self):
        prob = Problem(root=Group())
        root = prob.root
        root.add('p1', IndepVarComp('x', 1.0))
        root.add('c1', ExecComp('y = x*2.0'))
        root.add('c2', ExecComp('y = x*3.0'))
        root.connect('c1.x', 'c2.x')
        root.connect('p1.x', 'c2.x')
        prob.setup(check=False)
        prob.run()
        self.assertEqual(root.connections['c1.x'], ('p1.x', None))
        self.assertEqual(root.connections['c2.x'], ('p1.x', None))
        self.assertEqual(len(root.connections), 2)

    def test_explicit_connection_errors(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_state('x', 0)

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('x', 0)

        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('A', A())
        prob.root.add('B', B())

        prob.root.connect('A.x', 'B.x')
        prob.setup(check=False)

        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.y', 'B.x')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = ("Source 'A.y' cannot be connected to target 'B.x': "
                    "'A.y' does not exist.")
        self.assertEqual(str(cm.exception), expected)

        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.x', 'B.y')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = ("Source 'A.x' cannot be connected to target 'B.y': "
                    "'B.y' does not exist.")
        self.assertEqual(str(cm.exception), expected)

        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.x', 'A.x')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = ("Source 'A.x' cannot be connected to target 'A.x': "
                    "Target must be a parameter but 'A.x' is an unknown.")
        self.assertEqual(str(cm.exception), expected)

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

        # Type mismatch error message
        type_err = "Type <type '%s'> of source %s" \
                   " must be the same as "             \
                   "type <type '%s'> of target %s."

        # Type mismatch in explicit connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('E', E())

        prob.root.connect('A.y', 'E.y')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = py3fix(type_err % ('numpy.ndarray', "'A.y'", 'float', "'E.y'"))
        self.assertEqual(str(cm.exception), expected)

        # Type mismatch in implicit connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A(), promotes=['y'])
        prob.root.add('E', E(), promotes=['y'])

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = py3fix(type_err % ('numpy.ndarray', "'A.y' (y)", 'float', "'E.y' (y)"))
        self.assertEqual(str(cm.exception), expected)

        # Shape mismatch error message
        shape_err = "Shape %s of source %s" \
                    " must be the same as "     \
                    "shape %s of target %s."

        # Shape mismatch in explicit connection
        prob = Problem()
        prob.root = Group()

        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.y', 'B.y')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')

        expected = shape_err % ('(2,)', "'A.y'", '(3,)', "'B.y'")
        self.assertTrue(expected in raised_error)

        # Shape mismatch in implicit connection
        prob = Problem()
        prob.root = Group()

        prob.root.add('A', A(), promotes=['y'])
        prob.root.add('B', B(), promotes=['y'])

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')

        expected = shape_err % ('(2,)', "'A.y' (y)", '(3,)', "'B.y' (y)")
        self.assertTrue(expected in raised_error)

        # Shape mismatch in explicit connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('B', B())
        prob.root.add('C', C())
        prob.root.connect('C.y', 'B.y')

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')

        expected = shape_err % ('(2,)', "'C.y'", '(3,)', "'B.y'")
        self.assertTrue(expected in raised_error)

        # Shape mismatch in implicit connection
        prob = Problem()
        prob.root = Group()
        prob.root.add('B', B(), promotes=['y'])
        prob.root.add('C', C(), promotes=['y'])

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')

        expected = shape_err % ('(2,)', "'C.y' (y)", '(3,)', "'B.y' (y)")
        self.assertTrue(expected in raised_error)

        # Explicit
        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('A', A())
        prob.root.add('D', D())
        prob.root.connect('A.y', 'D.y')

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['no_unknown_comps'], ['D'])
        self.assertEqual(checks['recorders'], [])
        content = stream.getvalue()
        self.assertTrue("The following components have no unknowns:\nD\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

        # Implicit
        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('A', A(), promotes=['y'])
        prob.root.add('D', D(), promotes=['y'])

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['no_unknown_comps'], ['D'])
        self.assertEqual(checks['recorders'], [])
        content = stream.getvalue()
        self.assertTrue("The following components have no unknowns:\nD\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

        # Explicit
        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('C', C())
        prob.root.add('D', D())
        prob.root.connect('C.y', 'D.y')

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['no_unknown_comps'], ['D'])
        self.assertEqual(checks['recorders'], [])
        content = stream.getvalue()
        self.assertTrue("The following components have no unknowns:\nD\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

        # Implicit
        prob = Problem()
        prob.root = Group()
        prob.root.ln_solver = ScipyGMRES()
        prob.root.add('C', C(), promotes=['y'])
        prob.root.add('D', D(), promotes=['y'])

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['no_unknown_comps'], ['D'])
        self.assertEqual(checks['recorders'], [])
        content = stream.getvalue()
        self.assertTrue("The following components have no unknowns:\nD\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

    def test_src_idx_gt_src_size(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_state('y', np.zeros((2,)), shape=(2,))

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('x', np.zeros((3,)), shape=(3,))

        # src_indices larger than src
        prob = Problem(root=Group())
        prob.root.add("A", A())
        prob.root.add("B", B())
        prob.root.connect("A.y", "B.x", src_indices=[1,4,2])
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                    "'B.x' src_indices contains an index (4) that exceeds "
                    "the bounds of source variable 'A.y' of size 2.")
        else:
            self.fail("Exception expected")

    def test_src_idx_neg(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_output('y', np.zeros((5,)), shape=(5,))

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('x', np.zeros((3,)), shape=(3,))

        # src_indices larger than src
        prob = Problem(root=Group())
        prob.root.add("A", A())
        prob.root.add("B", B())
        prob.root.connect("A.y", "B.x", src_indices=[0, 1, -1])
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("'B.x' src_indices contains a negative index (-1)." in str(err))
        else:
            self.fail("Exception expected")

    def test_src_idx_gt_src_size(self):
        class A(Component):
            def __init__(self):
                super(A, self).__init__()
                self.add_output('y', np.zeros((5,)), shape=(5,))

        class B(Component):
            def __init__(self):
                super(B, self).__init__()
                self.add_param('x', np.zeros((3,)), shape=(3,))

        # src_indices larger than src
        prob = Problem(root=Group())
        prob.root.add("A", A())
        prob.root.add("B", B())
        prob.root.connect("A.y", "B.x", src_indices=[1,4,5])
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("'B.x' src_indices contains an index (5) that exceeds the bounds "
                             "of source variable 'A.y' of size 5." in str(err))
        else:
            self.fail("Exception expected")

    def test_simplest_run(self):

        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', IndepVarComp('x', 7.0))
        root.add('mycomp', ExecComp('y=x*2.0'))

        root.connect('x_param.x', 'mycomp.x')

        prob.setup(check=False)
        prob.run()
        result = root.unknowns['mycomp.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_simplest_run_w_promote(self):

        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', IndepVarComp('x', 7.0), promotes=['x'])
        root.add('mycomp', ExecComp('y=x*2.0'), promotes=['x'])

        prob.setup(check=False)
        prob.run()
        result = root.unknowns['mycomp.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_variable_access(self):
        prob = Problem(root=ExampleGroup())

        # set with a different shaped array
        try:
            prob['G2.C1.x']
        except Exception as err:
            msg = "'unknowns' has not been initialized, setup() must be called before 'G2.C1.x' can be accessed"
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

        prob.setup(check=False)

        self.assertEqual(prob['G2.C1.x'], 5.)                 # default output from IndepVarComp
        self.assertEqual(prob['G2.G1.C2.y'], 5.5)             # output from ExecComp
        self.assertEqual(prob.root.G3.C3.params['x'], 0.)     # initial value for a parameter
        self.assertEqual(prob.root.G2.G1.C2.params['x'], 0.)  # initial value for a parameter

        prob = Problem(root=ExampleGroupWithPromotes())
        prob.setup(check=False)
        self.assertEqual(prob.root.G2.G1.C2.params['x'], 0.)  # initial value for a parameter

        # __setitem__
        prob['G2.G1.C2.y'] = 99.
        self.assertEqual(prob['G2.G1.C2.y'], 99.)

    def test_variable_access_before_setup(self):
        prob = Problem(root=ExampleGroup())

        try:
            prob['G2.C1.x'] = 5.
        except AttributeError as err:
            msg = "'unknowns' has not been initialized, setup() must be called before 'G2.C1.x' can be accessed"
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

        try:
            prob.run()
        except RuntimeError as err:
            msg = "Before running the model, setup() must be called. If " + \
                "the configuration has changed since it was called, then " + \
                "setup must be called again before running the model."
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(prob['G3.C4.y'], 40.)

        stream = cStringIO()

        # get test coverage for list_connections and make sure it
        # doesn't barf
        prob.root.list_connections(stream=stream)
        prob.root.list_connections(unconnected=False, stream=stream)
        prob.root.list_connections(group_by_comp=False, stream=stream)
        prob.root.G3.C3.list_connections(var='x', stream=stream)

    def test_no_vecs(self):
        prob = Problem(root=ExampleGroup())
        prob.setup(check=False)

        # test that problem has no unknowns, params, etc.
        try:
            prob.unknowns['G3.C4.y']
        except AttributeError as err:
            self.assertEqual(str(err), "'Problem' object has no attribute 'unknowns'")
        else:
            self.fail("AttributeError expected")

        try:
            prob.params['G3.C4.x']
        except AttributeError as err:
            self.assertEqual(str(err), "'Problem' object has no attribute 'params'")
        else:
            self.fail("AttributeError expected")

        try:
            prob.resids['G3.C4.x']
        except AttributeError as err:
            self.assertEqual(str(err), "'Problem' object has no attribute 'resids'")
        else:
            self.fail("AttributeError expected")


    def test_byobj_run(self):
        prob = Problem(root=ExampleByObjGroup())

        prob.setup(check=False)
        prob.run()

        self.assertEqual(prob['G3.C4.y'], 'fooC2C3C4')

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
        root.add('X', IndepVarComp('x', 0., shape=1), promotes=['x'])
        root.add('A1', A(), promotes=['x'])
        root.add('A2', A())
        root.connect('A1.y', 'A2.x')
        prob.setup(check=False)

        # Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', IndepVarComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('B2', B())
        root.connect('B1.y', 'B2.x')
        prob.setup(check=False)

        # Mismatched Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', IndepVarComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('C1', C())
        root.connect('B1.y', 'C1.x')
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)
        expected = "Shape (2,) of source 'B1.y' must be the same as " \
                   "shape (3,) of target 'C1.x'."
        self.assertTrue(expected in str(cm.exception))

        # Mismatched Scalar to Array Value
        prob = Problem()
        root = prob.root = Group()
        root.add('X', IndepVarComp('x', 0., shape=1), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected = py3fix("Type <type 'float'> of source 'X.x' (x) must be the same as "
                          "type <type 'numpy.ndarray'> of target 'B1.x' (x).")
        self.assertEqual(expected, str(cm.exception))

    def test_mode_auto(self):
        # Make sure mode=auto chooses correctly for all prob sizes as well
        # as for abs/rel/etc paths

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('a', 1.0), promotes=['*'])
        root.add('p2', IndepVarComp('b', 1.0), promotes=['*'])
        root.add('comp', ExecComp(['x = 2.0*a + 3.0*b', 'y=4.0*a - 1.0*b']), promotes=['*'])

        root.ln_solver.options['mode'] = 'auto'
        prob.setup(check=False)
        prob.run()

        mode = prob._mode('auto', ['a'], ['x'])
        self.assertEqual(mode, 'fwd')

        mode = prob._mode('auto', ['a', 'b'], ['x'])
        self.assertEqual(mode, 'rev')

        # make sure _check function does it too

        #try:
            #mode = prob._check_for_parallel_derivs(['a'], ['x'], False, False)
        #except Exception as err:
            #msg  = "Group '' must have the same mode as root to use Matrix Matrix."
            #self.assertEqual(text_type(err), msg)
        #else:
            #self.fail('Exception expected')

        # Cheat a bit so I can twiddle mode
        OptionsDictionary.locked = False

        root.ln_solver.options['mode'] = 'fwd'

        mode = prob._check_for_parallel_derivs(['a', 'b'], ['x'], False, False)
        self.assertEqual(mode, 'fwd')

    def test_check_parallel_derivs(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('a', 1.0), promotes=['*'])
        root.add('p2', IndepVarComp('b', 1.0), promotes=['*'])
        sub1 = root.add('sub1', Group(), promotes=['*'])
        sub1.ln_solver = LinearGaussSeidel()
        sub2 = sub1.add('sub2', Group(), promotes=['*'])
        sub2.add('comp', ExecComp(['x = 2.0*a + 3.0*b', 'y=4.0*a - 1.0*b']), promotes=['*'])
        sub2.ln_solver = LinearGaussSeidel()

        root.ln_solver.options['mode'] = 'fwd'
        sub1.ln_solver.options['mode'] = 'fwd'
        sub2.ln_solver.options['mode'] = 'fwd'

        prob.setup(check=False)
        prob.run()

        root.ln_solver = LinearGaussSeidel()
        root.ln_solver.options['single_voi_relevance_reduction'] = True
        prob.driver.add_desvar('p1.a', 1.0)
        prob.driver.add_constraint('x', upper=0.0)
        prob.driver.add_constraint('y', upper=0.0)
        with warnings.catch_warnings(record=True) as w:
            if not MPI:
                # suppress warning about not running under MPI
                warnings.simplefilter("ignore")
            prob.driver.parallel_derivs(['x','y'])

        root.ln_solver.options['mode'] = 'rev'
        sub1.ln_solver.options['mode'] = 'rev'

        prob._setup_errors = []
        mode = prob._check_for_parallel_derivs(['a'], ['x'], True, False)

        msg  = "Group 'sub2' has mode 'fwd' but the root group has mode 'rev'. Modes must match to use parallel derivative groups."
        self.assertTrue(msg in prob._setup_errors[0])


        sub1.ln_solver.options['mode'] = 'fwd'
        sub2.ln_solver.options['mode'] = 'rev'


        prob._setup_errors = []
        mode = prob._check_for_parallel_derivs(['a'], ['x'], True, False)

        msg  = "Group 'sub1' has mode 'fwd' but the root group has mode 'rev'. Modes must match to use parallel derivative groups."
        self.assertTrue(msg in prob._setup_errors[0])


        sub1.ln_solver.options['mode'] = 'rev'
        mode = prob._check_for_parallel_derivs(['a'], ['x'], True, False)

    def test_iprint(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        base_stdout = sys.stdout

        try:
            ostream = cStringIO()
            sys.stdout = ostream
            top.run()
        finally:
            sys.stdout = base_stdout

        printed = ostream.getvalue()
        self.assertEqual(printed, '')

        # Turn on all iprints
        top.print_all_convergence()

        try:
            ostream = cStringIO()
            sys.stdout = ostream
            top.run()
        finally:
            sys.stdout = base_stdout

        printed = ostream.getvalue()
        self.assertEqual(printed.count('NEWTON'), 3)
        self.assertEqual(printed.count('GMRES'), 4)
        self.assertTrue('[root] NL: NEWTON   0 | ' in printed)
        self.assertTrue('   [root.sub] LN: GMRES   0 | ' in printed)

    def test_error_change_after_setup(self):

        # Tests error messages for the 5 options that we should never change
        # after setup is called.

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        # Not permitted to change this
        with self.assertRaises(RuntimeError) as err:
            top.root.fd_options['form'] = 'complex_step'

        expected_msg = "The 'form' option cannot be changed after setup."
        self.assertEqual(str(err.exception), expected_msg)

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        # Not permitted to change this
        with self.assertRaises(RuntimeError) as err:
            top.root.fd_options['extra_check_partials_form'] = 'complex_step'

        expected_msg = "The 'extra_check_partials_form' option cannot be changed after setup."
        self.assertEqual(str(err.exception), expected_msg)

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        # Not permitted to change this
        with self.assertRaises(RuntimeError) as err:
            top.root.fd_options['force_fd'] = True

        expected_msg = "The 'force_fd' option cannot be changed after setup."
        self.assertEqual(str(err.exception), expected_msg)

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        # Not permitted to change this
        with self.assertRaises(RuntimeError) as err:
            top.root.ln_solver.options['mode'] = 'rev'

        expected_msg = "The 'mode' option cannot be changed after setup."
        self.assertEqual(str(err.exception), expected_msg)

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        # Not permitted to change this
        with self.assertRaises(RuntimeError) as err:
            top.root.ln_solver.options['single_voi_relevance_reduction'] = True

        expected_msg = "The 'single_voi_relevance_reduction' option cannot be changed after setup."
        self.assertEqual(str(err.exception), expected_msg)

    def test_change_solver_after_setup(self):

        top = Problem()
        top.root = SellarStateConnection()
        top.setup(check=False)

        top.root.ln_solver = ScipyGMRES()

        with self.assertRaises(RuntimeError) as err:
            top.run()

        expected_msg = "Before running the model, setup() must be called. If " + \
            "the configuration has changed since it was called, then " + \
            "setup must be called again before running the model."
        self.assertEqual(str(err.exception), expected_msg)

class TestCheckSetup(unittest.TestCase):

    def test_out_of_order(self):
        prob = Problem(root=Group())
        root = prob.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group())
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", ExecComp('y=x*2.0'))
        C3 = G2.add("C3", ExecComp('y=x*2.0'))

        G2.connect("C1.y", "C3.x")
        G2.connect("C3.y", "C2.x")

        # force wrong order
        G2.set_order(['C1', 'C2', 'C3'])

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)
        self.assertEqual(checks['out_of_order'], [('G1.G2',[('C2',['C3'])])])

    def test_cycle(self):
        prob = Problem(root=Group())
        root = prob.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group())
        G2.ln_solver = ScipyGMRES()

        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", ExecComp('y=x*2.0'))
        C3 = G2.add("C3", ExecComp('y=x*2.0'))

        G2.connect("C1.y", "C3.x")
        G2.connect("C3.y", "C2.x")
        G2.connect("C2.y", "C1.x")

        # force wrong order
        G2.set_order(['C1', 'C2', 'C3'])

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)
        auto, _ = G2.list_auto_order()
        self.assertTrue(auto==['C1', 'C3', 'C2'] or
                        auto==['C3', 'C2', 'C1'] or
                        auto==['C2', 'C1', 'C3'])
        self.assertTrue("Group 'G1.G2' has the following cycles: [['C1', 'C2', 'C3']]" in
                        stream.getvalue())

        oo = checks['out_of_order']
        self.assertEqual(oo[0][0], 'G1.G2')
        expected = {
            ('C2','C3'): 'C1',
            ('C3',): 'C2',
            ('C2',): 'C1',
        }

        for node, afters in oo[0][1]:
            self.assertEqual(node, expected[tuple(afters)])

    def test_pbo_messages(self):

        class PBOSrcComp(Component):

            def __init__(self):
                super(PBOSrcComp, self).__init__()

                self.add_param('x1', 100.0)
                self.add_output('x2', 100.0, units='degC', pass_by_obj=True)
                self.fd_options['force_fd'] = True

            def solve_nonlinear(self, params, unknowns, resids):
                """ No action."""
                unknowns['x2'] = params['x1']

        class PBOTgtCompF(Component):

            def __init__(self):
                super(PBOTgtCompF, self).__init__()

                self.add_param('x2', 100.0, units='degF', pass_by_obj=True)
                self.add_output('x3', 100.0)
                self.fd_options['force_fd'] = True

            def solve_nonlinear(self, params, unknowns, resids):
                """ No action."""
                unknowns['x3'] = params['x2']

        # Don't warn for driver that doesn't need gradients.
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', PBOSrcComp())
        prob.root.add('tgtF', PBOTgtCompF())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['relevant_pbos'], [])

        class GradDriver(Driver):
            def __init__(self):
                super(GradDriver, self).__init__()
                self.supports['gradients'] = True

        # Do warn with a gradient driver
        prob = Problem()
        prob.root = Group()
        prob.root.add('src', PBOSrcComp())
        prob.root.add('tgtF', PBOTgtCompF())
        prob.root.add('tgtF2', PBOTgtCompF())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtF2.x2')
        prob.driver = GradDriver()

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)

        self.assertEqual(checks['relevant_pbos'], ['src.x2'])

if __name__ == "__main__":
    unittest.main()

""" Unit test for the Problem class. """

import unittest
import numpy as np
from six import text_type, PY3
from six.moves import cStringIO
import warnings

from openmdao.components.linear_system import LinearSystem
from openmdao.core.component import Component
from openmdao.core.problem import Problem
from openmdao.core.checks import ConnectError
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.test.examplegroups import ExampleGroup, ExampleGroupWithPromotes, ExampleByObjGroup
from openmdao.test.simplecomps import SimpleComp, SimpleImplicitComp, RosenSuzuki, FanIn


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
        G2.add('C1', ParamComp('x', 5.), promotes=['x'])

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
            msg = "Target 'G3.C4.x' is connected to multiple unknowns: ['G3.C3.y', 'G2.C1.x']"
            self.assertEqual(text_type(error), msg)
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
            msg = "'C' promotes 'spoon' but has no variables matching that specification"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

        # promoting a pattern with no matches should throw an error
        prob = Problem(Group())
        G = prob.root.add('G', Group())
        P = G.add('P', ParamComp('x', 5.), promotes=['a*'])     # there is no match
        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "'P' promotes 'a*' but has no variables matching that specification"
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
        G2.add('C1', ParamComp('x', 5.), promotes=['x'])

        G1 = G2.add('G1', Group(), promotes=['x'])
        G1.add('C2', ExecComp('y=x*2.0'), promotes=['x'])

        G3 = root.add('G3', Group(), promotes=['x'])
        G3.add('C3', ExecComp('y=x*2.0'), promotes=['y'])          # promoting y
        G3.add('C4', ExecComp('y=x*2.0'), promotes=['x', 'y'])     # promoting y again.. BAD

        prob = Problem(root)

        try:
            prob.setup(check=False)
        except Exception as error:
            msg = "Promoted name 'G3.y' matches multiple unknowns: ['G3.C3.y', 'G3.C4.y']"
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

        expected_msg = "Promoted name 'z' matches multiple unknowns: ['c1.z', 'c2.z']"

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
        root.add('p1', ParamComp('x', 1.0))
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
        root.add('p1', ParamComp('x', 1.0))
        root.add('c1', ExecComp('y = x*2.0'))
        root.add('c2', ExecComp('y = x*3.0'))
        root.connect('c1.x', 'c2.x')
        root.connect('p1.x', 'c2.x')
        prob.setup(check=False)
        prob.run()
        self.assertEqual(root.connections['c1.x'], 'p1.x')
        self.assertEqual(root.connections['c2.x'], 'p1.x')
        self.assertEqual(len(root.connections), 2)

    def test_calc_gradient_interface_errors(self):

        root = Group()
        prob = Problem(root=root)
        root.add('comp', ExecComp('y=x*2.0'))

        try:
            prob.calc_gradient(['comp.x'], ['comp.y'], mode='junk')
        except Exception as error:
            msg = "mode must be 'auto', 'fwd', 'rev', or 'fd'"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

        try:
            prob.calc_gradient(['comp.x'], ['comp.y'], return_format='junk')
        except Exception as error:
            msg = "return_format must be 'array' or 'dict'"
            self.assertEqual(text_type(error), msg)
        else:
            self.fail("Error expected")

    def test_calc_gradient(self):
        root = Group()
        parm = root.add('parm', ParamComp('x', np.array([1., 1., 1., 1.])))
        comp = root.add('comp', RosenSuzuki())

        root.connect('parm.x', 'comp.x')

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        param_list = ['parm.x']
        unknown_list = ['comp.f', 'comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]))
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp.f']['parm.x'], np.array([
            [ -3., -3., -17.,  9.],
        ]), decimal=5)
        np.testing.assert_almost_equal(J['comp.g']['parm.x'], np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([
            [-3.,  -3., -17.,  9.],
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ]), decimal=5)

    def test_calc_gradient_multiple_params(self):
        prob = Problem()
        prob.root = FanIn()
        prob.setup(check=False)
        prob.run()

        param_list   = ['p1.x1', 'p2.x2']
        unknown_list = ['comp3.y']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fwd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(param_list, unknown_list, mode='rev', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fd', return_format='dict')
        np.testing.assert_almost_equal(J['comp3.y']['p2.x2'], np.array([[ 35.]]))
        np.testing.assert_almost_equal(J['comp3.y']['p1.x1'], np.array([[ -6.]]))

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(param_list, unknown_list, mode='fd', return_format='array')
        np.testing.assert_almost_equal(J, np.array([[-6., 35.]]))

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
        prob.root.add('A', A())
        prob.root.add('B', B())

        prob.root.connect('A.x', 'B.x')
        prob.setup(check=False)

        expected_error_message = ("Source 'A.y' cannot be connected to target 'B.x': "
                                  "'A.y' does not exist.")
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.y', 'B.x')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected_error_message)

        expected_error_message = ("Source 'A.x' cannot be connected to target 'B.y': "
                                  "'B.y' does not exist.")
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.x', 'B.y')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected_error_message)

        expected_error_message = ("Source 'A.x' cannot be connected to target 'A.x': "
                                  "Target must be a parameter but 'A.x' is an unknown.")
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.x', 'A.x')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected_error_message)

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
                                  "'A.y' must be the same as type "
                                  "'<type 'float'>' of target "
                                  "'E.y'")
        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A())
        prob.root.add('E', E())

        prob.root.connect('A.y', 'E.y')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected_error_message)

        #Implicit
        expected_error_message = py3fix("Type '<type 'numpy.ndarray'>' of source "
                                  "'y' must be the same as type "
                                  "'<type 'float'>' of target "
                                  "'y'")

        prob = Problem()
        prob.root = Group()
        prob.root.add('A', A(), promotes=['y'])
        prob.root.add('E', E(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        self.assertEqual(str(cm.exception), expected_error_message)


        # Explicit
        expected_error_message = ("Shape '(2,)' of the source 'A.y' "
                                  "must match the shape '(3,)' "
                                  "of the target 'B.y'")
        prob = Problem()
        prob.root = Group()

        prob.root.add('A', A())
        prob.root.add('B', B())
        prob.root.connect('A.y', 'B.y')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')
        self.assertEqual(raised_error, expected_error_message)

        # Implicit
        expected_error_message = ("Shape '(2,)' of the source 'y' "
                                  "must match the shape '(3,)' "
                                  "of the target 'y'")

        prob = Problem()
        prob.root = Group()

        prob.root.add('A', A(), promotes=['y'])
        prob.root.add('B', B(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')
        self.assertEqual(raised_error, expected_error_message)

        # Explicit
        expected_error_message = ("Shape '(2,)' of the source 'C.y' must match the shape '(3,)' "
                                  "of the target 'B.y'")

        prob = Problem()
        prob.root = Group()
        prob.root.add('B', B())
        prob.root.add('C', C())
        prob.root.connect('C.y', 'B.y')

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')
        self.assertEqual(raised_error, expected_error_message)

        # Implicit
        expected_error_message = ("Shape '(2,)' of the source 'y' must match the shape"
                                  " '(3,)' of the target 'y'")

        prob = Problem()
        prob.root = Group()
        prob.root.add('B', B(), promotes=['y'])
        prob.root.add('C', C(), promotes=['y'])

        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        raised_error = str(cm.exception)
        raised_error = raised_error.replace('(2L,', '(2,')
        raised_error = raised_error.replace('(3L,', '(3,')
        self.assertEqual(raised_error, expected_error_message)

        # Explicit
        prob = Problem()
        prob.root = Group()
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
        prob.root.add('C', C(), promotes=['y'])
        prob.root.add('D', D(), promotes=['y'])
        stream = cStringIO()
        checks = prob.setup(out_stream=stream)
        self.assertEqual(checks['no_unknown_comps'], ['D'])
        self.assertEqual(checks['recorders'], [])
        content = stream.getvalue()
        self.assertTrue("The following components have no unknowns:\nD\n" in content)
        self.assertTrue("No recorders have been specified, so no data will be saved." in content)

    def test_simplest_run(self):

        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', ParamComp('x', 7.0))
        root.add('mycomp', ExecComp('y=x*2.0'))

        root.connect('x_param.x', 'mycomp.x')

        prob.setup(check=False)
        prob.run()
        result = root.unknowns['mycomp.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_simplest_run_w_promote(self):

        prob = Problem(root=Group())
        root = prob.root

        # ? Didn't we say that ParamComp by default promoted its variable?
        root.add('x_param', ParamComp('x', 7.0), promotes=['x'])
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

        self.assertEqual(prob['G2.C1.x'], 5.)                 # default output from ParamComp
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
        except AttributeError as err:
            msg = "'unknowns' has not been initialized, setup() must be called before 'x' can be accessed"
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(prob['G3.C4.y'], 40.)

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
        root.add('X', ParamComp('x', 0., shape=1), promotes=['x'])
        root.add('A1', A(), promotes=['x'])
        root.add('A2', A())
        root.connect('A1.y', 'A2.x')
        prob.setup(check=False)

        # Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('B2', B())
        root.connect('B1.y', 'B2.x')
        prob.setup(check=False)

        # Mismatched Array Values
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', np.zeros(2), shape=2), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        root.add('C1', C())
        root.connect('B1.y', 'C1.x')
        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)
        expected_error_message = "Shape '(2,)' of the source "\
                                  "'B1.y' must match the shape '(3,)' "\
                                  "of the target 'C1.x'"
        self.assertEqual(expected_error_message, str(cm.exception))

        # Mismatched Scalar to Array Value
        prob = Problem()
        root = prob.root = Group()
        root.add('X', ParamComp('x', 0., shape=1), promotes=['x'])
        root.add('B1', B(), promotes=['x'])
        with self.assertRaises(ConnectError) as cm:
            prob.setup(check=False)

        expected_error_message = py3fix("Type '<type 'float'>' of source "
                                  "'x' must be the same as type "
                                  "'<type 'numpy.ndarray'>' of target "
                                  "'x'")
        self.assertEqual(expected_error_message, str(cm.exception))

    def test_mode_auto(self):
        # Make sure mode=auto chooses correctly for all prob sizes as well
        # as for abs/rel/etc paths

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('a', 1.0), promotes=['*'])
        root.add('p2', ParamComp('b', 1.0), promotes=['*'])
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
            #mode = prob._check_for_matrix_matrix(['a'], ['x'])
        #except Exception as err:
            #msg  = "Group '' must have the same mode as root to use Matrix Matrix."
            #self.assertEqual(text_type(err), msg)
        #else:
            #self.fail('Exception expected')

        root.ln_solver.options['mode'] = 'fwd'
        mode = prob._check_for_matrix_matrix(['a', 'b'], ['x'])
        self.assertEqual(mode, 'fwd')

    def test_check_matrix_matrix(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', ParamComp('a', 1.0), promotes=['*'])
        root.add('p2', ParamComp('b', 1.0), promotes=['*'])
        sub1 = root.add('sub1', Group(), promotes=['*'])
        sub2 = sub1.add('sub2', Group(), promotes=['*'])
        sub2.add('comp', ExecComp(['x = 2.0*a + 3.0*b', 'y=4.0*a - 1.0*b']), promotes=['*'])

        prob.setup(check=False)
        prob.run()

        # NOTE: this call won't actually calculate mode because default ln_solver
        # is ScipyGMRES and its default mode is 'fwd', not 'auto'.
        mode = prob._check_for_matrix_matrix(['a'], ['x'])

        root.ln_solver.options['mode'] = 'rev'
        sub1.ln_solver.options['mode'] = 'rev'

        try:
            mode = prob._check_for_matrix_matrix(['a'], ['x'])
        except Exception as err:
            msg  = "Group 'sub2' has mode 'fwd' but the root group has mode 'rev'. Modes must match to use Matrix Matrix."
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

        sub1.ln_solver.options['mode'] = 'fwd'
        sub2.ln_solver.options['mode'] = 'rev'

        try:
            mode = prob._check_for_matrix_matrix(['a'], ['x'])
        except Exception as err:
            msg  = "Group 'sub1' has mode 'fwd' but the root group has mode 'rev'. Modes must match to use Matrix Matrix."
            self.assertEqual(text_type(err), msg)
        else:
            self.fail('Exception expected')

        sub1.ln_solver.options['mode'] = 'rev'
        mode = prob._check_for_matrix_matrix(['a'], ['x'])


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

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)
        self.assertEqual(checks['out_of_order'], [('G1.G2',[('C2',['C3'])])])

    def test_cycle(self):
        prob = Problem(root=Group())
        root = prob.root

        G1 = root.add("G1", Group())
        G2 = G1.add("G2", Group())
        C1 = G2.add("C1", ExecComp('y=x*2.0'))
        C2 = G2.add("C2", ExecComp('y=x*2.0'))
        C3 = G2.add("C3", ExecComp('y=x*2.0'))

        G2.connect("C1.y", "C3.x")
        G2.connect("C3.y", "C2.x")
        G2.connect("C2.y", "C1.x")

        stream = cStringIO()
        checks = prob.setup(out_stream=stream)
        self.assertTrue("Group 'G1.G2' has the following cycles: [['C1', 'C2', 'C3']]" in
                        stream.getvalue())
        self.assertEqual(checks['out_of_order'], [('G1.G2',[('C2',['C3'])])])


if __name__ == "__main__":
    unittest.main()

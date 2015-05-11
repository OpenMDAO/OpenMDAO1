""" Unit test for the Problem class. """

import unittest
from six import text_type

from openmdao.components.linear_system import LinearSystem
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp
from openmdao.test.examplegroups import ExampleGroup, ExampleGroupWithPromotes

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
        self.assertEqual(prob['G2:G1:x', 'params'], 0.)      # initial value for a parameter

    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        prob.setup()
        prob.run()

        self.assertAlmostEqual(prob['G3:C4:y'], 40.)


if __name__ == "__main__":
    unittest.main()

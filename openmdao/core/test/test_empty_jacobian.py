""" Test to verify that components without outputs will work as expected as
long as their linearize method returns an empty dict {}.
"""
import unittest

from openmdao.api import Problem, Group, Component, ExecComp, IndepVarComp
from openmdao.drivers.scipy_optimizer import ScipyOptimizer


class InputComp(Component):

    def __init__(self):

        super(InputComp, self).__init__()

        self.add_param(name="A", val=0.0)
        self.add_param(name="B", val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def linearize(self, params, unknowns, resids):
        return {}


class InputComp2(Component):

    def __init__(self):

        super(InputComp2, self).__init__()

        self.add_param(name="A", val=0.0)
        self.add_param(name="B", val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def linearize(self, params, unknowns, resids):
        return None


class InputComp3(Component):

    def __init__(self):

        super(InputComp3, self).__init__()

        self.add_param(name="A", val=0.0)
        self.add_param(name="B", val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        pass


class MyGroup(Group):

    def __init__(self):
        super(MyGroup, self).__init__()

        self.add(name="input", system=InputComp())

        self.add(name="exec", system=ExecComp("y=a**2+b**2", a=5.0, b=5.0))

        self.connect("input.A", "exec.a")
        self.connect("input.B", "exec.b")


class MyGroup2(Group):

    def __init__(self):
        super(MyGroup2, self).__init__()

        self.add(name="input", system=InputComp2())

        self.add(name="exec", system=ExecComp("y=a**2+b**2", a=5.0, b=5.0))

        self.connect("input.A", "exec.a")
        self.connect("input.B", "exec.b")


class MyGroup3(Group):

    def __init__(self):
        super(MyGroup3, self).__init__()

        self.add(name="input", system=InputComp3())

        self.add(name="exec", system=ExecComp("y=a**2+b**2", a=5.0, b=5.0))

        self.connect("input.A", "exec.a")
        self.connect("input.B", "exec.b")


class TestEmptyJacobian(unittest.TestCase):

    def test_empty_jacobian(self):

        prob = Problem(root=Group())

        root = prob.root

        root.add(name="ivc_a", system=IndepVarComp(name="a", val=5.0),
                 promotes=["a"])
        root.add(name="ivc_b", system=IndepVarComp(name="b", val=10.0),
                 promotes=["b"])

        root.add(name="g", system=MyGroup())

        root.connect("a", "g.input.A")
        root.connect("b", "g.input.B")

        prob.driver = ScipyOptimizer()
        prob.driver.options["disp"] = False
        prob.driver.add_objective("g.exec.y")
        prob.driver.add_desvar(name="a")
        prob.driver.add_desvar(name="b")

        prob.setup(check=False)

        prob.run()


    def test_none_jacobian(self):

        prob = Problem(root=Group())

        root = prob.root

        root.add(name="ivc_a", system=IndepVarComp(name="a", val=5.0),
                 promotes=["a"])
        root.add(name="ivc_b", system=IndepVarComp(name="b", val=10.0),
                 promotes=["b"])

        root.add(name="g", system=MyGroup2())

        root.connect("a", "g.input.A")
        root.connect("b", "g.input.B")

        prob.driver = ScipyOptimizer()
        prob.driver.options["disp"] = False
        prob.driver.add_objective("g.exec.y")
        prob.driver.add_desvar(name="a")
        prob.driver.add_desvar(name="b")

        prob.setup(check=False)

        try:
            prob.run()
        except ValueError as err:
            self.assertEqual(str(err), "No derivatives defined for Component 'g.input'")
        else:
            self.fail("expecting ValueError due to linearize returning None")


    def test_undefined_jacobian(self):

        prob = Problem(root=Group())

        root = prob.root

        root.add(name="ivc_a", system=IndepVarComp(name="a", val=5.0),
                 promotes=["a"])
        root.add(name="ivc_b", system=IndepVarComp(name="b", val=10.0),
                 promotes=["b"])

        root.add(name="g", system=MyGroup3())

        root.connect("a", "g.input.A")
        root.connect("b", "g.input.B")

        prob.driver = ScipyOptimizer()
        prob.driver.options["disp"] = False
        prob.driver.add_objective("g.exec.y")
        prob.driver.add_desvar(name="a")
        prob.driver.add_desvar(name="b")

        prob.setup(check=False)

        try:
            prob.run()
        except ValueError as err:
            self.assertEqual(str(err), "No derivatives defined for Component 'g.input'")
        else:
            self.fail("expecting ValueError due to undefined linearize")


if __name__ == "__main__":
    unittest.main()

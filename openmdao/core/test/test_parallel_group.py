import unittest

from openmdao.api import ParallelGroup, Problem, IndepVarComp, ExecComp, NLGaussSeidel


class TestGroup(unittest.TestCase):

    def setUp(self):
        root = ParallelGroup()

        root.nl_solver = NLGaussSeidel()

        root.add('C1', IndepVarComp('x', 5.))
        root.add('C2', ExecComp('y=x*2.0'))
        root.add('C3', ExecComp('y=x*2.0'))
        root.add('C4', ExecComp('y=x*2.0'))

        root.connect("C1.x", "C2.x")
        root.connect("C2.y", "C4.x")
        root.connect("C4.y", "C3.x")

        self.root = root

    def test_run(self):

        prob = Problem(self.root)
        prob.setup(check=False)
        prob.run()

        self.assertEqual(self.root.nl_solver.iter_count, 3)
        self.assertEqual(prob['C3.y'], 40.)

    def test_list_auto_order(self):
        prob = Problem(self.root)
        prob.setup(check=False)

        self.assertEqual(self.root.list_auto_order(),
                         (['C1', 'C2', 'C3', 'C4'],[]))

if __name__ == "__main__":
    unittest.main()

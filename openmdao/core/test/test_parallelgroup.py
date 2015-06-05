import unittest

from openmdao.core.problem import Problem
from openmdao.core.parallelgroup import ParallelGroup
from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel


class TestGroup(unittest.TestCase):

    def test_run(self):

        prob = Problem(root=ParallelGroup())
        root = prob.root
        root.nl_solver = NLGaussSeidel()

        root.add('C1', ParamComp('x', 5.))
        root.add('C2', ExecComp('y=x*2.0'))
        root.add('C3', ExecComp('y=x*2.0'))
        root.add('C4', ExecComp('y=x*2.0'))

        root.connect("C1:x", "C2:x")
        root.connect("C2:y", "C3:x")
        root.connect("C3:y", "C4:x")

        prob.setup()
        prob.run()

        self.assertEqual(root.nl_solver.iter_count, 3)
        self.assertEqual(prob['C4:y'], 40.)

if __name__ == "__main__":
    unittest.main()

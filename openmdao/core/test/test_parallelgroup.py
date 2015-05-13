import unittest

from openmdao.core.problem import Problem
from openmdao.core.parallelgroup import ParallelGroup
from openmdao.components.paramcomp import ParamComp
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.test.simplecomps import SimpleComp


class TestGroup(unittest.TestCase):

    def test_run(self):

        prob = Problem(root=ParallelGroup())
        root = prob.root
        root.nl_solver = NLGaussSeidel()

        root.add('C1', ParamComp('x', 5.))
        root.add('C2', SimpleComp())
        root.add('C3', SimpleComp())
        root.add('C4', SimpleComp())

        root.connect("C1:x", "C2:x")
        root.connect("C2:y", "C3:x")
        root.connect("C3:y", "C4:x")

        prob.setup()
        prob.run()

        self.assertEqual(root.nl_solver.iter_count, 3)
        self.assertEqual(prob['C4:y'], 40.)

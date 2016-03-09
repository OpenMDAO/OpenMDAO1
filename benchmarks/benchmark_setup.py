
import unittest

from openmdao.api import Problem, Group
from openmdao.test.test_builder import DynComp, create_dyncomps


class BenchmarkCompVars(unittest.TestCase):
    """Some tests for setup of a component with a large
    number of variables.
    """

    def _build_comp(self, np, no, ns=0):
        prob = Problem(root=Group())
        prob.root.add("C1", DynComp(np, no, ns))
        return prob

    def benchmark_1000params(self):
        prob = self._build_comp(1000, 1)
        prob.setup(check=False)

    def benchmark_2000params(self):
        prob = self._build_comp(2000, 1)
        prob.setup(check=False)

    def benchmark_1000outs(self):
        prob = self._build_comp(1, 1000)
        prob.setup(check=False)

    def benchmark_2000outs(self):
        prob = self._build_comp(1, 2000)
        prob.setup(check=False)

    def benchmark_1000vars(self):
        prob = self._build_comp(500, 500)
        prob.setup(check=False)

    def benchmark_2000vars(self):
        prob = self._build_comp(1000, 1000)
        prob.setup(check=False)

class BenchmarkManySystems(unittest.TestCase):
    """Setup of models with lots of systems"""

    def benchmark_flat_100_systems(self):
        p = Problem(root=Group())
        create_dyncomps(p.root, 100, 20, 5)
        p.setup(check=False)

    def benchmark_flat_1000_systems(self):
        p = Problem(root=Group())
        create_dyncomps(p.root, 1000, 20, 5)
        p.setup(check=False)

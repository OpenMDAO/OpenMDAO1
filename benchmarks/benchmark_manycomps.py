
import unittest

from openmdao.api import Problem, Group
from openmdao.test.build4test import DynComp, create_dyncomps

class BenchmarkManyComps(unittest.TestCase):
    """Setup of models with lots of components"""

    def benchmark_flat_100_comps(self):
        p = Problem(root=Group())
        create_dyncomps(p.root, 100, 10, 10, 5)
        p.setup(check=False)

    def benchmark_flat_500_comps(self):
        p = Problem(root=Group())
        create_dyncomps(p.root, 500, 10, 10, 5)
        p.setup(check=False)

    def benchmark_flat_1000_comps(self):
        p = Problem(root=Group())
        create_dyncomps(p.root, 1000, 10, 10, 5)
        p.setup(check=False)

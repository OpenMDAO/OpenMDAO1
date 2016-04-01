
import unittest

from openmdao.api import Problem, Group
from openmdao.devtools.debug import stats
from openmdao.test.build4test import DynComp, create_dyncomps, make_subtree

class BM(unittest.TestCase):
    """Setup of models with various tree structures"""

    def benchmark_L4_sub2_c10(self):
        p = Problem(root=Group())
        make_subtree(p.root, nsubgroups=2, levels=4, ncomps=10,
                     nparams=10, noutputs=10, nconns=5)
        p.setup(check=False)

    def benchmark_L6_sub2_c10(self):
        p = Problem(root=Group())
        make_subtree(p.root, nsubgroups=2, levels=6, ncomps=10,
                     nparams=10, noutputs=10, nconns=5)
        p.setup(check=False)

    def benchmark_L7_sub2_c10(self):
        p = Problem(root=Group())
        make_subtree(p.root, nsubgroups=2, levels=7, ncomps=10,
                     nparams=10, noutputs=10, nconns=5)
        p.setup(check=False)

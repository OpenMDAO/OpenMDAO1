""" Testing group-level finite difference. """

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.paramcomp import ParamComp
from openmdao.components.execcomp import ExecComp
from openmdao.test.testutil import assert_rel_error


class TestIndices(unittest.TestCase):

    def test_indices(self):
        size = 10

        root = Group()

        root.add('P1', ParamComp('x', np.zeros(size)))
        root.add('C1', ExecComp('y = x * 2.', y=np.zeros(size//2), x=np.zeros(size//2)))
        root.add('C2', ExecComp('y = x * 3.', y=np.zeros(size//2), x=np.zeros(size//2)))

        root.connect('P1.x', "C1.x", src_indices=list(range(size//2)))
        root.connect('P1.x', "C2.x", src_indices=list(range(size//2, size)))

        prob = Problem(root)
        prob.setup(check=False)

        root.P1.unknowns['x'][0:size//2] += 1.0
        root.P1.unknowns['x'][size//2:size] -= 1.0

        prob.run()

        assert_rel_error(self, root.C1.params['x'], np.ones(size//2), 0.0001)
        assert_rel_error(self, root.C2.params['x'], -np.ones(size//2), 0.0001)


if __name__ == "__main__":
    unittest.main()

""" Testing group-level finite difference. """

import unittest

import numpy as np

from openmdao.core import Group, Problem
from openmdao.components import IndepVarComp, ExecComp

from openmdao.test.simple_comps import SimpleComp, SimpleArrayComp
from openmdao.test.util import assert_rel_error


class TestIndices(unittest.TestCase):

    def test_indices(self):
        size = 10

        root = Group()

        root.add('P1', IndepVarComp('x', np.zeros(size)))
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

    def test_array_to_scalar(self):
        root = Group()

        root.add('P1', IndepVarComp('x', np.array([2., 3.])))
        root.add('C1', SimpleComp())
        root.add('C2', ExecComp('y = x * 3.', y=0., x=0.))

        root.connect('P1.x', 'C1.x', src_indices=[0,])
        root.connect('P1.x', 'C2.x', src_indices=[1,])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(root.C1.params['x'], 2.)
        self.assertAlmostEqual(root.C2.params['x'], 3.)

    def test_subarray_to_promoted_var(self):
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A = G.add('A', SimpleArrayComp()) # , promotes=['x', 'y'])

        root.connect('P.x', 'G.A.x', src_indices=[0,1])
        root.connect('P.x', 'C.x', src_indices=[2,])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, root.G.A.params['x'], np.array([1., 2.]), 0.0001)
        self.assertAlmostEqual(root.C.params['x'], 3.)

        # no try the same thing with promoted var
        root = Group()

        P = root.add('P', IndepVarComp('x', np.array([1., 2., 3.])))
        G = root.add('G', Group())
        C = root.add('C', SimpleComp())

        A = G.add('A', SimpleArrayComp(), promotes=['x', 'y'])

        root.connect('P.x', 'G.x', src_indices=[0,1])
        root.connect('P.x', 'C.x', src_indices=[2,])

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        assert_rel_error(self, root.G.A.params['x'], np.array([1., 2.]), 0.0001)
        self.assertAlmostEqual(root.C.params['x'], 3.)


if __name__ == "__main__":
    unittest.main()

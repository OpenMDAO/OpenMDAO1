# Test the LinearSystem component.

import unittest
import numpy as np

from openmdao.api import Group, Problem, LinearSystem, IndepVarComp, ScipyGMRES
from openmdao.test.util import assert_rel_error


class TestLinearSystem(unittest.TestCase):

    def test_linear_system(self):
        root = Group()

        lingrp = root.add('lingrp', Group(), promotes=['*'])
        lingrp.add('lin', LinearSystem(3))
        lingrp.ln_solver = ScipyGMRES()

        x = np.array([1, 2, -3])
        A = np.array([[ 5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x)

        root.add('p1', IndepVarComp('A', A))
        root.add('p2', IndepVarComp('b', b))
        root.connect('p1.A', 'lin.A')
        root.connect('p2.b', 'lin.b')

        prob = Problem(root)
        prob.setup(check=False)
        prob.run()

        # Make sure it gets the right answer
        assert_rel_error(self, prob['lin.x'], x, .0001)
        assert_rel_error(self, np.linalg.norm(prob.root.resids.vec), 0.0, 1e-10)

        # Compare against calculated derivs
        Ainv = np.linalg.inv(A)
        dx_dA = np.outer(Ainv, -x).reshape(3, 9)
        dx_db = Ainv

        J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)

        J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='rev', return_format='dict')
        assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)

        J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='fd', return_format='dict')
        assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)


if __name__ == "__main__":
    unittest.main()

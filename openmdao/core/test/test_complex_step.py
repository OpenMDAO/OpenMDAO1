""" Testing out complex step capability."""

from __future__ import print_function

import unittest

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.test.paraboloid import Paraboloid


class ComplexStepVectorUnitTests(unittest.TestCase):

    def test_single_comp_paraboloid(self):
        prob = Problem()
        root = prob.root = Group()
        root.add('p1', IndepVarComp('x', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 0.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        root.fd_options['force_fd'] = True
        root.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        jac = prob.calc_gradient(['x', 'y'], ['f_xy'])
        print(jac)

if __name__ == "__main__":
    unittest.main()
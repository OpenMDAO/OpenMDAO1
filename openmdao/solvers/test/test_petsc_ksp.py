""" This tets out the Petsc KSP solver in Serial mode. """

import unittest
import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.param_comp import ParamComp
from openmdao.solvers.petsc_ksp import PetscKSP
from openmdao.test.simple_comps import SimpleCompDerivMatVec

from openmdao.core.petsc_impl import PetscImpl as impl


class TestPetscKSPSerial(unittest.TestCase):

    def test_simple(self):
        group = Group()
        group.add('x_param', ParamComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)


if __name__ == "__main__":
    unittest.main()

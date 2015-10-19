""" Testing out complex step capability."""

import unittest

import numpy as np

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.vec_wrapper_complex_step import ComplexStepSrcVecWrapper, \
                                                   ComplexStepTgtVecWrapper
from openmdao.test.simple_comps import ArrayComp2D

class ComplexStepVectorUnitTests(unittest.TestCase):

    def test_param_vec(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ArrayComp2D(), promotes=['x', 'y'])
        top.root.add('p1', IndepVarComp('x', np.array([[1.0, 2.0], [3.0, 4.0]])),
                     promotes=['x'])

        top.setup(check=False)
        top.run()

        params = ComplexStepTgtVecWrapper(top.root.comp.params)

        # Get a param that isn't complex-stepped
        x = params['x']
        self.assertTrue(x.dtype == np.float)
        self.assertTrue(x[0, 1] == 2.0)

        # Get a param that is now complex
        params.set_complex_var('x')
        x = params['x']
        self.assertTrue(x.dtype == np.complex)
        self.assertTrue(x[0, 1] == 2.0 + 0j)

        # Apply complex step and get param
        params.step_complex(1, 4.0)
        self.assertTrue(x[0, 1] == 2.0 + 4j)

    def test_unknown_vec(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ArrayComp2D(), promotes=['x', 'y'])
        top.root.add('p1', IndepVarComp('x', np.array([[1.0, 2.0], [3.0, 4.0]])),
                     promotes=['x'])

        top.setup(check=False)
        top.run()

        unknowns = ComplexStepSrcVecWrapper(top.root.comp.unknowns)

        # Unknowns are always complex
        y = unknowns['y']
        self.assertTrue(y.dtype == np.complex)
        self.assertTrue(y[0, 1] == 46.0 + 0j)

        # Set an unknown
        y[0, 1]= 13.0 + 17.0j
        self.assertTrue(y[0, 1] == 13.0 + 17.0j)

        # Extract flat var


if __name__ == "__main__":
    unittest.main()

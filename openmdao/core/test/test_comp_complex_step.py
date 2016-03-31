""" Testing out complex step capability."""

import unittest

import numpy as np

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.test.test_comp_fd_jacobian import TestProb
from openmdao.core.test.test_units import SrcComp, TgtCompC, TgtCompF, TgtCompK
from openmdao.core.vec_wrapper_complex_step import ComplexStepSrcVecWrapper, \
                                                   ComplexStepTgtVecWrapper
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.simple_comps import ArrayComp2D, SimpleComp
from openmdao.test.util import assert_rel_error


class ComplexStepVectorUnitTests(unittest.TestCase):

    def test_param_vec(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ArrayComp2D(), promotes=['x', 'y'])
        top.root.add('p1', IndepVarComp('x', np.array([[1.0, 2.0], [3.0, 4.0]])),
                     promotes=['x'])
        top.root.add('comp2', SimpleComp())
        top.root.add('p2', IndepVarComp('x', 3.0))
        top.root.connect('p2.x', 'comp2.x')

        top.setup(check=False)
        top.run()

        params = ComplexStepTgtVecWrapper(top.root.comp.params)

        # Get a param that isn't complex-stepped
        x = params['x']
        self.assertTrue(x.dtype == np.float)
        self.assertEquals(x[0, 1], 2.0)

        # Get a param that is now complex
        params.set_complex_var('x')
        x = params['x']
        self.assertTrue(x.dtype == np.complex)
        self.assertEquals(x[0, 1], 2.0 + 0j)

        # Apply complex step and get param
        params.step_complex(1, 4.0)
        x = params['x']
        self.assertEquals(x[0, 1], 2.0 + 4j)

        # Unset complex
        params.set_complex_var(None)
        x = params['x']
        self.assertEquals(x[0, 1], 2.0)

        params = ComplexStepTgtVecWrapper(top.root.comp2.params)

        # Get a param that isn't complex-stepped
        x = params['x']
        self.assertTrue(x.dtype == np.float)
        self.assertEquals(x, 3.0)

        # Get a param that is now complex
        params.set_complex_var('x')
        x = params['x']
        self.assertTrue(x.dtype == np.complex)
        self.assertEquals(x, 3.0 + 0j)

        # Apply complex step and get param
        params.step_complex(0, 4.0)
        self.assertEquals(x, 3.0 + 4j)

        # Make sure all other functions work for coverage
        self.assertEquals(len(params), 1)
        self.assertTrue('x' in params)
        plist = [z for z in params]
        self.assertEquals(plist, ['x'])
        self.assertEquals(params.keys(), top.root.comp2.params.keys())
        self.assertEquals(params.metadata('x'), top.root.comp2.params.metadata('x'))
        plist1 = [z for z in params.iterkeys()]
        plist2 = [z for z in top.root.comp2.params.iterkeys()]

    def test_unknown_vec(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ArrayComp2D(), promotes=['x', 'y'])
        top.root.add('p1', IndepVarComp('x', np.array([[1.0, 2.0], [3.0, 4.0]])),
                     promotes=['x'])
        top.root.add('comp2', SimpleComp())
        top.root.add('p2', IndepVarComp('x', 3.0))
        top.root.connect('p2.x', 'comp2.x')

        top.setup(check=False)
        top.run()

        unknowns = ComplexStepSrcVecWrapper(top.root.comp.unknowns)

        # Unknowns are always complex
        y = unknowns['y']
        self.assertTrue(y.dtype == np.complex)
        self.assertEquals(y[0, 1], 46.0 + 0j)

        # Set an unknown
        unknowns['y'][0, 1]= 13.0 + 17.0j
        self.assertEquals(unknowns['y'][0, 1], 13.0 + 17.0j)

        # Extract flat var
        cval = unknowns.flat('y')
        self.assertEquals(cval[1], 13.0 + 17.0j)
        self.assertEquals(cval.shape[0], 4)

        unknowns = ComplexStepSrcVecWrapper(top.root.comp2.unknowns)

        # Unknowns are always complex
        y = unknowns['y']
        self.assertTrue(y.dtype == np.complex)
        self.assertEquals(y, 6.0 + 0j)

        # Set an unknown
        unknowns['y'] = 13.0 + 17.0j
        self.assertEquals(unknowns['y'], 13.0 + 17.0j)

        # Extract flat var
        cval = unknowns.flat('y')
        self.assertEquals(cval, 13.0 + 17.0j)
        self.assertEquals(cval.shape[0], 1)

        # Make sure all other functions work for coverage
        self.assertEquals(len(unknowns), 1)
        self.assertTrue('y' in unknowns)
        plist = [z for z in unknowns]
        self.assertEquals(plist, ['y'])
        self.assertEquals(unknowns.keys(), top.root.comp2.unknowns.keys())
        self.assertEquals(unknowns.metadata('y'), top.root.comp2.unknowns.metadata('y'))
        plist1 = [z for z in unknowns.iterkeys()]
        plist2 = [z for z in top.root.comp2.unknowns.iterkeys()]

    def test_unit_convert(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgtF', TgtCompF())
        prob.root.add('tgtC', TgtCompC())
        prob.root.add('tgtK', TgtCompK())
        prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src.x1')
        prob.root.connect('src.x2', 'tgtF.x2')
        prob.root.connect('src.x2', 'tgtC.x2')
        prob.root.connect('src.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run()

        p1 = ComplexStepTgtVecWrapper(prob.root.tgtF.params)
        p2 = ComplexStepTgtVecWrapper(prob.root.tgtC.params)
        p3 = ComplexStepTgtVecWrapper(prob.root.tgtK.params)

        assert_rel_error(self, p1['x2'], 212.0, 1.0e-6)
        assert_rel_error(self, p2['x2'], 100.0, 1.0e-6)
        assert_rel_error(self, p3['x2'], 373.15, 1.0e-6)


class ComplexStepComponentTests(unittest.TestCase):

    def test_simple_float(self):

        prob = Problem()
        prob.root = root = Group()
        root.add('x_param', IndepVarComp('x', 17.0), promotes=['x'])
        root.add('y_param', IndepVarComp('y', 19.0), promotes=['y'])
        root.add('mycomp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        # This will give poor FD, but good CS
        root.mycomp.fd_options['step_size'] = 1.0e1
        root.mycomp.fd_options['force_fd'] = True
        root.mycomp.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['f_xy'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['f_xy']['x'][0][0], 47.0, 1e-6)

    def test_array2D(self):

        prob = Problem()
        prob.root = root = Group()
        root.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        root.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        root.mycomp.fd_options['step_size'] = 1.0e-1
        root.mycomp.fd_options['force_fd'] = True
        root.mycomp.fd_options['form'] = 'complex_step'

        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = prob.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_override_states(self):

        expected_keys=[('y', 'x'), ('y', 'z'), ('z', 'x'), ('z', 'z')]

        p = TestProb()
        p.setup(check=False)

        params = p.root.ci1.params
        unknowns = p.root.ci1.unknowns
        resids = p.root.ci1.resids

        jac = p.root.ci1.fd_jacobian(params, unknowns, resids)
        self.assertEqual(set(expected_keys), set(jac.keys()))

        # Don't compute derivs wrt 'z'
        expected_keys=[('y', 'x'), ('z', 'x')]

        params = p.root.ci1.params
        unknowns = p.root.ci1.unknowns
        resids = p.root.ci1.resids

        jac = p.root.ci1.fd_jacobian(params, unknowns, resids, fd_states=[])
        self.assertEqual(set(expected_keys), set(jac.keys()))

if __name__ == "__main__":
    unittest.main()

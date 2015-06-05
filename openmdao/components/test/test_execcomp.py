import unittest
from unittest import SkipTest
import math
from six import iteritems

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.execcomp import ExecComp
from openmdao.test.testutil import assert_rel_error


class TestExecComp(unittest.TestCase):

    def test_simple(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp('y=x+1.', x=2.0))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], 3.0, 0.00001)

    def test_with_derivs(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp(['y=2.0*x+1.'], ['dy_dx=2.0'], x=2.0))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], 5.0, 0.00001)

        J = C1.jacobian(C1.params, C1.unknowns, C1.resids)

        assert_rel_error(self, J[('y','x')], 2.0, 0.00001)

    def test_math(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp('y=sin(x)', x=2.0))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], math.sin(2.0), 0.00001)

    def test_array(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp('y=x[1]', x=np.array([1.,2.,3.]), y=0.0))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], 2.0, 0.00001)

    def test_array_lhs(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp(['y[0]=x[1]', 'y[1]=x[0]'],
                                       x=np.array([1.,2.,3.]), y=np.array([0.,0.])))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], np.array([2.,1.]), 0.00001)

    def test_simple_array_model(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ExecComp(['y[0]=2.0*x[0]+7.0*x[1]',
                                       'y[1]=5.0*x[0]-3.0*x[1]'],
                                      ['dy_dx=numpy.array([[2.0,7.0],[5.0,-3.0]])'],
                                      x=np.zeros([2]), y=np.zeros([2])))

        top.root.add('p1', ParamComp('x', np.ones([2])))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        data = top.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_array_model2(self):

        top = Problem()
        top.root = Group()
        top.root.add('comp', ExecComp(['y[0]=2.0*x[0]+7.0*x[1]',
                                       'y[1]=5.0*x[0]-3.0*x[1]'],
                                      ['dy_dx[0,0]=2.0',
                                       'dy_dx[0,1]=7.0',
                                       'dy_dx[1,0]=5.0',
                                       'dy_dx[1,1]=-3.0'],
                                      x=np.zeros([2]), y=np.zeros([2]),
                                      dy_dx=np.zeros((2,2))))

        top.root.add('p1', ParamComp('x', np.ones([2])))

        top.root.connect('p1:x', 'comp:x')

        top.setup()
        top.run()

        data = top.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_complex_step(self):
        p = Problem(root=Group())
        C1 = p.root.add('C1', ExecComp(['y=2.0*x+1.'], x=2.0))
        self.assertTrue('x' in C1._params_dict)
        self.assertTrue('y' in C1._unknowns_dict)

        p.setup()
        p.run()

        assert_rel_error(self, C1.unknowns['y'], 5.0, 0.00001)

        J = C1.jacobian(C1.params, C1.unknowns, C1.resids)

        assert_rel_error(self, J[('y','x')], 2.0, 0.00001)



    def test_complex_step2(self):

        top = Problem()
        top.root = Group()
        comp = top.root.add('comp', ExecComp('y=x*x + x*2.0'))
        top.root.add('p1', ParamComp('x', 2.0))
        top.root.connect('p1:x', 'comp:x')

        comp.fd_options['force_fd'] = False

        top.setup()
        top.run()

        J = top.calc_gradient(['comp:x'], ['comp:y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp:y']['comp:x'], np.array([6.0]), 0.00001)

